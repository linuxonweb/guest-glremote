/* glremote: guest OpenGL ES 3.0 + EGL 1.5 over gpu_hostcall to the host
 * browser's WebGL2 (see glremote.h, ops.def and dist/guest-gl-remote.js).
 *
 * Wire format: OP_GL_EXEC carries a stream of 32-bit words; each command is a
 * header (op | nwords << 12) followed by nwords argument words. Floats travel
 * as their bits. A data blob is [bytes, mode, ...]: mode 0 inline words, 1 null,
 * 2 guest pointer (the batch is flushed before the call returns), 3 offset into
 * the bound pixel/element buffer. Only the last command of a batch may be a
 * query; its reply goes to the context reply buffer (OP_GL_EXEC returns its
 * length, or the needed length when it did not fit).
 *
 * State the GL client can answer itself (bindings, viewport, enables, pixel
 * store, limits, program introspection after link) is kept here so the common
 * save/restore queries of GL helpers do not wait for the host. */
#define _GNU_SOURCE
#define GL_GLES_PROTOTYPES 0
#define EGL_EGL_PROTOTYPES 0
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <errno.h>
#include <malloc.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

#include "glremote.h"

#define GPU_HC_CALL 0x6a470001L
#define HC_GL_OPEN 0x30
#define HC_GL_EXEC 0x31
#define HC_GL_CLOSE 0x32
#define OPEN_PROBE 1u
#define OPEN_ALPHA 2u
#define OPEN_DEPTH 4u
#define OPEN_STENCIL 8u

#define GL_HALF_FLOAT_OES 0x8D61
#define GL_BGRA_EXT 0x80E1

#define CMD_WORDS (256u * 1024u)
#define INLINE_MAX (64u * 1024u)
/* Larger inputs are copied and sent by pointer with the next batch; past this
 * many copied bytes the batch is sent early. */
#define STAGE_MAX (64u * 1024u * 1024u)
/* Guest copies of buffer contents (see struct gbuf). */
#define SHADOW_MAX (16u * 1024u * 1024u)
#define SHADOW_BUDGET (96u * 1024u * 1024u)
#define INFO_CAP (64u * 1024u)
#define MAX_UNITS 96
#define MAX_VALUES 1024

enum {
  OP_NONE,
#define OP(name) OP_##name,
#include "ops.def"
#undef OP
  OP_COUNT
};
static const char *const op_names[] = {
  "other", /* profile: batches sent without a query or pointer */
#define OP(name) #name,
#include "ops.def"
#undef OP
};
enum { K_BUFFER, K_TEXTURE, K_FRAMEBUFFER, K_RENDERBUFFER, K_QUERY, K_SAMPLER, K_VERTEX_ARRAY, K_TRANSFORM_FEEDBACK, K_PROGRAM, K_SHADER, K_COUNT };
enum { B_INLINE, B_NULL, B_POINTER, B_OFFSET };

/* ---- diagnostics ---- */

static int trace_level = -1;
static int tracing(void) {
  if (trace_level < 0) {
    const char *t = getenv("GLREMOTE_TRACE");
    trace_level = t && *t && strcmp(t, "0") ? (atoi(t) > 0 ? atoi(t) : 1) : 0;
  }
  return trace_level;
}
static void say(const char *format, ...) {
  char line[1024];
  va_list args;
  va_start(args, format);
  int n = vsnprintf(line, sizeof line, format, args);
  va_end(args);
  if (n < 0) return;
  if (n >= (int)sizeof line) n = sizeof line - 1;
  fprintf(stderr, "GLREMOTE %.*s\n", n, line);
}

static long hostcall(long op, long a0, long a1, long a2, long a3, long a4) {
  long r = syscall(GPU_HC_CALL, op, a0, a1, a2, a3, a4);
  return r < 0 ? -errno : r;
}

int glremote_available(void) {
  static int state = -1;
  if (state < 0) state = hostcall(HC_GL_OPEN, OPEN_PROBE, 0, 0, 0, 0) == 0;
  return state;
}

/* ---- context state ---- */

enum { PS_UNPACK_ALIGNMENT, PS_UNPACK_ROW_LENGTH, PS_UNPACK_IMAGE_HEIGHT, PS_UNPACK_SKIP_PIXELS, PS_UNPACK_SKIP_ROWS,
       PS_UNPACK_SKIP_IMAGES, PS_PACK_ALIGNMENT, PS_PACK_ROW_LENGTH, PS_PACK_SKIP_PIXELS, PS_PACK_SKIP_ROWS, PS_COUNT };
static const GLenum ps_names[PS_COUNT] = {
  GL_UNPACK_ALIGNMENT, GL_UNPACK_ROW_LENGTH, GL_UNPACK_IMAGE_HEIGHT, GL_UNPACK_SKIP_PIXELS, GL_UNPACK_SKIP_ROWS,
  GL_UNPACK_SKIP_IMAGES, GL_PACK_ALIGNMENT, GL_PACK_ROW_LENGTH, GL_PACK_SKIP_PIXELS, GL_PACK_SKIP_ROWS,
};
static const GLenum caps[] = {
  GL_BLEND, GL_CULL_FACE, GL_DEPTH_TEST, GL_DITHER, GL_POLYGON_OFFSET_FILL, GL_PRIMITIVE_RESTART_FIXED_INDEX,
  GL_RASTERIZER_DISCARD, GL_SAMPLE_ALPHA_TO_COVERAGE, GL_SAMPLE_COVERAGE, GL_SCISSOR_TEST, GL_STENCIL_TEST,
};
#define NCAPS (sizeof caps / sizeof caps[0])

struct var { const char *name; GLint size, loc, block, offset, astride, mstride, rowmajor; GLenum type; };
struct block { const char *name; GLint binding, datasize, refvs, reffs, n; GLint *indices; };
struct obj { /* program or shader */
  int is_program;
  GLenum type;
  char *source;
  int info_valid;
  char *text;
  GLint status, validate, deleted, log_len, tfmode, nattached;
  GLuint attached[8];
  const char *log;
  int na, nu, nb, nv;
  struct var *a, *u, *v;
  struct block *b;
};
struct limit { GLenum pname; int n; double *v; };
/* A buffer's contents as last sent to the host. Write-only maps that keep
 * the old contents (WebRender maps without an invalidate bit) start from this
 * copy instead of reading the buffer back, which takes a host round trip.
 * The copy is dropped when the host itself writes the buffer (ReadPixels into
 * a pack buffer, CopyBufferSubData); buffers ever bound for transform
 * feedback never get one. */
struct gbuf { unsigned char *shadow; size_t size; int host_written; };
struct mapping { GLenum target; GLuint buffer; void *ptr; GLintptr offset; GLsizeiptr length; GLbitfield access; };
struct glr_sync { uint32_t magic; struct glr_ctx *ctx; };
#define SYNC_MAGIC 0x676c7273u

struct glr_surface;
struct glr_ctx {
  int32_t host;
  /* Random bytes the host finds at this address while the context lives; it
   * reclaims host contexts of guests that exited without destroying them. */
  uint32_t cookie[2];
  int lost, bound, destroy_pending, made_current;
  struct glr_ctx *next;
  struct glr_surface *draw, *read;
  /* Host canvas size, and the window surface whose frame it holds (for
   * EGL_EXT_buffer_age); resizing the canvas clears it. */
  EGLint canvas_w, canvas_h;
  struct glr_surface *shown;
  /* The host offers keyed presentation ("present keyed"); the surface the
   * context's key belongs to (one window per host canvas). */
  int keyed;
  struct glr_surface *key_surface;
  EGLint config_id;
  uint32_t *cmd, used, since_error;
  int must_flush;
  /* Input copies referenced by the unsent batch, freed once it ran. */
  void **staged;
  size_t nstaged, staged_cap, staged_bytes;
  /* Large blocks kept for the next buffer mapping instead of a fresh mmap. */
  void *spare[4];
  /* Framebuffer completeness: bumped by every command that can change it;
   * the last host answer holds while the count and the binding are the same. */
  uint32_t fb_gen, fbs_gen;
  GLuint fbs_name;
  GLenum fbs_value;
  uint64_t execs, words, queries;
  /* GLREMOTE_TRACE=2: a summary every 5 s of where batches end. */
  struct glr_prof {
    uint64_t start_ns, host_ns, words, execs, pointer_flushes, full_flushes;
    uint32_t query_ops[OP_COUNT], flush_ops[OP_COUNT], error_after[OP_COUNT], host_us[OP_COUNT];
    unsigned end_op, last_op;
    uint64_t swaps, swap_ns, window_ns, swap_pixels, swap_damaged, swap_exposed, swap_first, buffer_age_queries;
  } prof;
  void *reply;
  uint32_t reply_cap;
  GLenum error;
  /* host info */
  char *info;
  const char *version, *renderer, *vendor, *glsl, *extensions;
  const char **ext;
  int n_ext;
  struct limit *limits;
  int nlimits;
  /* names */
  GLuint next_name[K_COUNT];
  struct obj **objs;
  size_t nobjs;
  GLuint scratch_elements;
  /* tracked state */
  GLint ps[PS_COUNT];
  unsigned unit;
  GLuint tex[MAX_UNITS][4], sampler[MAX_UNITS];
  GLuint array_buffer, copy_read, copy_write, pack_buffer, unpack_buffer, uniform_buffer, tf_buffer;
  struct gbuf *bufs;
  size_t nbufs, shadow_bytes;
  GLuint draw_fb, read_fb, renderbuffer, program, vao, tfo;
  GLuint *vao_elements;
  size_t nvao;
  GLint viewport[4], scissor[4];
  unsigned char enabled[NCAPS];
  struct mapping maps[8];
};

static _Thread_local struct glr_ctx *tls_ctx;

static void add_error(struct glr_ctx *c, GLenum e) {
  if (!c->error) c->error = e;
}

/* ---- command stream ---- */

static uint64_t now_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

/* Appends the most frequent entries of `counts` (zeroing them) to `line`. */
static int top_ops(char *line, int n, int cap, uint32_t *counts) {
  for (int pass = 0; pass < 8; pass++) {
    int best = 0;
    for (int op = 1; op < OP_COUNT; op++) if (counts[op] > counts[best]) best = op;
    if (!counts[best] || n >= cap - 48) break;
    n += snprintf(line + n, (size_t)(cap - n), " %s=%u", op_names[best], counts[best]);
    counts[best] = 0;
  }
  return n;
}

static void prof_report(struct glr_ctx *c, uint64_t now) {
  struct glr_prof *p = &c->prof;
  char line[1024];
  int n = top_ops(line, 0, (int)sizeof line, p->query_ops);
  n += snprintf(line + n, sizeof line - (size_t)n, " | pointer flushes:");
  n = top_ops(line, n, (int)sizeof line, p->flush_ops);
  n += snprintf(line + n, sizeof line - (size_t)n, " | GetError after:");
  n = top_ops(line, n, (int)sizeof line, p->error_after);
  n += snprintf(line + n, sizeof line - (size_t)n, " | host us by batch end:");
  n = top_ops(line, n, (int)sizeof line, p->host_us);
  line[n] = 0;
  double secs = (double)(now - p->start_ns) / 1e9;
  say("prof ctx=%d %.1fs execs=%llu host=%.0fms words=%llu pointer-flushes=%llu full-flushes=%llu swaps=%llu swap=%.0fms "
      "window=%.0fms swap-pixels=%llu damaged=%llu exposed=%llu unpresented=%llu age-queries=%llu queries:%s", c->host, secs,
      (unsigned long long)p->execs, (double)p->host_ns / 1e6, (unsigned long long)p->words,
      (unsigned long long)p->pointer_flushes, (unsigned long long)p->full_flushes, (unsigned long long)p->swaps,
      (double)p->swap_ns / 1e6, (double)p->window_ns / 1e6, (unsigned long long)p->swap_pixels,
      (unsigned long long)p->swap_damaged, (unsigned long long)p->swap_exposed, (unsigned long long)p->swap_first,
      (unsigned long long)p->buffer_age_queries, line);
  memset(p, 0, sizeof *p);
  p->start_ns = now;
}

/* Mapping and staging blocks of INLINE_MAX bytes or more are mmap-backed in
 * musl's malloc; a few freed ones (64 MiB at most) are reused instead. */
static void *block_alloc(struct glr_ctx *c, size_t n) {
  for (int i = 0; n >= INLINE_MAX && i < 4; i++) {
    size_t have = c->spare[i] ? malloc_usable_size(c->spare[i]) : 0;
    if (have >= n && have - n <= n / 2 + INLINE_MAX) {
      void *p = c->spare[i];
      c->spare[i] = NULL;
      return p;
    }
  }
  return malloc(n);
}
static void block_release(struct glr_ctx *c, void *p) {
  size_t n = p ? malloc_usable_size(p) : 0, kept = n;
  int slot = -1;
  for (int i = 0; i < 4; i++) {
    if (!c->spare[i]) slot = slot < 0 ? i : slot;
    else kept += malloc_usable_size(c->spare[i]);
  }
  if (n >= INLINE_MAX && slot >= 0 && kept <= 64u * 1024u * 1024u) c->spare[slot] = p;
  else free(p);
}

static long run(struct glr_ctx *c, void *reply, uint32_t cap) {
  long r;
  uint64_t t0 = tracing() >= 2 ? now_ns() : 0;
  if (c->lost) r = -EIO;
  else r = hostcall(HC_GL_EXEC, c->host, (long)(uintptr_t)c->cmd, (long)c->used * 4, (long)(uintptr_t)reply, (long)cap);
  if (t0) {
    uint64_t t1 = now_ns();
    if (!c->prof.start_ns) c->prof.start_ns = t0;
    c->prof.host_ns += t1 - t0;
    c->prof.host_us[c->prof.end_op] += (uint32_t)((t1 - t0) / 1000);
    c->prof.end_op = 0;
    c->prof.words += c->used;
    c->prof.execs++;
    if (t1 - c->prof.start_ns >= 5000000000u) prof_report(c, t1);
  }
  c->execs++;
  c->words += c->used;
  c->used = 0;
  c->must_flush = 0;
  for (size_t i = 0; i < c->nstaged; i++) block_release(c, c->staged[i]);
  c->nstaged = 0;
  c->staged_bytes = 0;
  if (r < 0 && !c->lost) {
    c->lost = 1;
    say("context %d lost: exec failed errno %ld", c->host, -r);
  }
  if (tracing() >= 3) say("exec ctx=%d -> %ld", c->host, r);
  return r;
}

static void flush(struct glr_ctx *c) {
  if (c->used) run(c, NULL, 0);
}

static uint32_t *emit(struct glr_ctx *c, unsigned op, uint32_t n) {
  if (c->used + 1 + n > CMD_WORDS) {
    c->prof.full_flushes++;
    flush(c);
  }
  uint32_t *w = c->cmd + c->used;
  if (op == OP_GetError) c->prof.error_after[c->prof.last_op]++;
  c->prof.last_op = op;
  w[0] = op | (n << 12);
  c->used += 1 + n;
  c->since_error++;
  switch (op) {
  case OP_FramebufferTexture2D: case OP_FramebufferTextureLayer: case OP_FramebufferRenderbuffer:
  case OP_TexImage2D: case OP_TexImage3D: case OP_CompressedTexImage2D: case OP_CompressedTexImage3D:
  case OP_CopyTexImage2D: case OP_TexStorage2D: case OP_TexStorage3D: case OP_TexParameteri: case OP_TexParameterf:
  case OP_RenderbufferStorage: case OP_RenderbufferStorageMultisample: case OP_GenerateMipmap:
  case OP_DeleteObjects: case OP_SetSurface:
    c->fb_gen++;
  }
  return w + 1;
}

static uint32_t fbits(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}

#define CMD(op, ...) do { \
    const uint32_t a_[] = {__VA_ARGS__}; \
    memcpy(emit(c, OP_##op, sizeof a_ / 4), a_, sizeof a_); \
  } while (0)
#define CMD0(op) ((void)emit(c, OP_##op, 0))

/* Fixed arguments followed by one blob; returns the fixed-argument words. */
/* Keeps `p` (malloc'd) until the current batch has run. */
static int stage(struct glr_ctx *c, void *p, size_t bytes) {
  if (c->nstaged == c->staged_cap) {
    size_t cap = c->staged_cap ? c->staged_cap * 2 : 16;
    void **grown = realloc(c->staged, cap * sizeof *grown);
    if (!grown) return 0;
    c->staged = grown;
    c->staged_cap = cap;
  }
  c->staged[c->nstaged++] = p;
  c->staged_bytes += bytes;
  return 1;
}

/* Input data (B_INLINE) beyond INLINE_MAX is copied and sent by pointer with
 * the batch, so the caller need not wait for the host. */
static uint32_t *emit_data(struct glr_ctx *c, unsigned op, uint32_t nfixed, int mode, const void *data, size_t bytes) {
  if (mode == B_INLINE && !data) mode = B_NULL;
  int big = mode == B_INLINE && bytes > INLINE_MAX;
  if (big) {
    mode = B_POINTER;
    if (c->staged_bytes + bytes > STAGE_MAX) flush(c);
  }
  uint32_t extra = mode == B_INLINE ? (uint32_t)((bytes + 3) / 4) : mode == B_NULL ? 0 : 1;
  uint32_t *w = emit(c, op, nfixed + 2 + extra); /* may run the batch: stage after it */
  w[nfixed] = (uint32_t)bytes;
  w[nfixed + 1] = (uint32_t)mode;
  if (mode == B_INLINE) {
    if (extra) w[nfixed + 1 + extra] = 0;
    memcpy(w + nfixed + 2, data, bytes);
  } else if (mode != B_NULL) {
    const void *sent = data;
    if (big) {
      void *copy = block_alloc(c, bytes);
      if (copy && stage(c, copy, bytes)) {
        memcpy(copy, data, bytes);
        sent = copy;
      } else {
        free(copy);
        c->must_flush = 1; /* send before returning, while `data` is valid */
        c->prof.flush_ops[op]++;
        c->prof.end_op = op;
      }
    } else if (mode == B_POINTER) {
      c->must_flush = 1; /* output pointers (ReadPixels) */
      c->prof.flush_ops[op]++;
      c->prof.end_op = op;
    }
    w[nfixed + 2] = (uint32_t)(uintptr_t)sent;
  }
  return w;
}

/* Ends every call that may have sent a guest pointer. */
static void done(struct glr_ctx *c) {
  if (c->must_flush) {
    c->prof.pointer_flushes++;
    flush(c);
  }
}

static uint32_t *emit_query(struct glr_ctx *c, unsigned op, uint32_t n, uint32_t *start) {
  if (c->used + 1 + n > CMD_WORDS) flush(c);
  *start = c->used;
  return emit(c, op, n);
}

/* Runs the batch ending with the query at `start`; the reply length or -1. */
static long query(struct glr_ctx *c, uint32_t start) {
  c->queries++;
  c->prof.query_ops[c->cmd[start] & 0xfffu]++;
  c->prof.end_op = c->cmd[start] & 0xfffu;
  uint32_t n = c->used - start;
  long r = run(c, c->reply, c->reply_cap);
  if (r > (long)c->reply_cap && !c->lost) {
    /* The query alone again, with room for the reply (queries are idempotent). */
    uint32_t cap = (uint32_t)r + 4096;
    void *bigger = realloc(c->reply, cap);
    if (!bigger) return -1;
    c->reply = bigger;
    c->reply_cap = cap;
    memmove(c->cmd, c->cmd + start, (size_t)n * 4); /* the words are still in the buffer */
    c->used = n;
    r = run(c, c->reply, c->reply_cap);
  }
  return r < 0 || r > (long)c->reply_cap ? -1 : r;
}

/* Numeric replies: [u32 count][u32 0][f64 values]. Returns the count or -1. */
static int query_numbers(struct glr_ctx *c, uint32_t start, double *out, int max) {
  long r = query(c, start);
  if (r < 8) return -1;
  uint32_t count;
  memcpy(&count, c->reply, 4);
  if (8 + (long)count * 8 > r) return -1;
  int n = (int)count < max ? (int)count : max;
  memcpy(out, (char *)c->reply + 8, (size_t)n * 8);
  return (int)count;
}

static int q2(struct glr_ctx *c, unsigned op, uint32_t a0, uint32_t a1, int nargs, double *out, int max) {
  uint32_t start;
  uint32_t *w = emit_query(c, op, (uint32_t)nargs, &start);
  if (nargs > 0) w[0] = a0;
  if (nargs > 1) w[1] = a1;
  return query_numbers(c, start, out, max);
}
static int q3(struct glr_ctx *c, unsigned op, uint32_t a0, uint32_t a1, uint32_t a2, double *out, int max) {
  uint32_t start;
  uint32_t *w = emit_query(c, op, 3, &start);
  w[0] = a0; w[1] = a1; w[2] = a2;
  return query_numbers(c, start, out, max);
}

/* ---- objects ---- */

static void gen_names(struct glr_ctx *c, int kind, GLsizei n, GLuint *names) {
  if (n < 0) { add_error(c, GL_INVALID_VALUE); return; }
  if (!n || !names) return;
  uint32_t *w = emit(c, OP_GenObjects, 2 + (uint32_t)n);
  w[0] = (uint32_t)kind;
  w[1] = (uint32_t)n;
  for (GLsizei i = 0; i < n; i++) w[2 + i] = names[i] = ++c->next_name[kind];
}

static struct obj *object(struct glr_ctx *c, GLuint name) {
  return name && name < c->nobjs ? c->objs[name] : NULL;
}
static struct obj *new_object(struct glr_ctx *c, GLuint name, int is_program, GLenum type) {
  if (name >= c->nobjs) {
    size_t n = c->nobjs ? c->nobjs : 64;
    while (n <= name) n *= 2;
    struct obj **grown = realloc(c->objs, n * sizeof *grown);
    if (!grown) return NULL;
    memset(grown + c->nobjs, 0, (n - c->nobjs) * sizeof *grown);
    c->objs = grown;
    c->nobjs = n;
  }
  struct obj *o = calloc(1, sizeof *o);
  if (!o) return NULL;
  o->is_program = is_program;
  o->type = type;
  c->objs[name] = o;
  return o;
}
static void clear_info(struct obj *o) {
  for (int i = 0; i < o->nb; i++) free(o->b[i].indices);
  free(o->a); free(o->u); free(o->v); free(o->b); free(o->text);
  o->a = o->u = o->v = NULL;
  o->b = NULL;
  o->text = NULL;
  o->na = o->nu = o->nv = o->nb = 0;
  o->log = NULL;
  o->log_len = 0;
  o->nattached = 0;
  o->info_valid = 0;
}

static int split(char *line, char **fields, int max) {
  int n = 0;
  for (char *p = line; *p && n < max;) {
    while (*p == ' ') p++;
    if (!*p) break;
    fields[n++] = p;
    while (*p && *p != ' ') p++;
    if (*p) *p++ = 0;
  }
  return n;
}

/* Program and shader introspection text, see guest-gl-remote.js programInfo(). */
static int load_info(struct glr_ctx *c, GLuint name, struct obj *o) {
  if (o->info_valid) return 1;
  clear_info(o);
  uint32_t start;
  emit_query(c, o->is_program ? OP_ProgramInfo : OP_ShaderInfo, 1, &start)[0] = name;
  long len = query(c, start);
  if (len < 0) return 0;
  char *text = malloc((size_t)len + 1);
  if (!text) return 0;
  memcpy(text, c->reply, (size_t)len);
  text[len] = 0;
  o->text = text;
  char *p = text, *end = text + len;
  while (p < end) {
    char *nl = memchr(p, '\n', (size_t)(end - p));
    if (!nl) nl = end;
    *nl = 0;
    char *line = p;
    p = nl + 1;
    char *f[16];
    int nf = split(line, f, 16);
    if (nf < 2) continue;
    if (!strcmp(f[0], "log")) {
      long n = atol(f[1]);
      if (n < 0 || n > end - p) n = p <= end ? end - p : 0;
      o->log = p;
      o->log_len = (GLint)n;
      if (p + n < end) p[n] = 0;
      p += n + 1;
    } else if (!strcmp(f[0], "link") || !strcmp(f[0], "compile")) {
      o->status = atoi(f[1]);
    } else if (!strcmp(f[0], "validate")) {
      o->validate = atoi(f[1]);
    } else if (!strcmp(f[0], "deleted")) {
      o->deleted = atoi(f[1]);
    } else if (!strcmp(f[0], "tf")) {
      o->tfmode = atoi(f[1]);
    } else if (!strcmp(f[0], "attached")) {
      for (int i = 1; i < nf && o->nattached < 8; i++) o->attached[o->nattached++] = (GLuint)strtoul(f[i], NULL, 10);
    } else if (!strcmp(f[0], "a") && nf >= 5) {
      struct var *grown = realloc(o->a, (size_t)(o->na + 1) * sizeof *grown);
      if (!grown) continue;
      o->a = grown;
      o->a[o->na++] = (struct var){.size = atoi(f[1]), .type = (GLenum)atoi(f[2]), .loc = atoi(f[3]), .name = f[4], .block = -1};
    } else if (!strcmp(f[0], "u") && nf >= 9) {
      struct var *grown = realloc(o->u, (size_t)(o->nu + 1) * sizeof *grown);
      if (!grown) continue;
      o->u = grown;
      o->u[o->nu++] = (struct var){.size = atoi(f[1]), .type = (GLenum)atoi(f[2]), .block = atoi(f[3]), .offset = atoi(f[4]),
                                   .astride = atoi(f[5]), .mstride = atoi(f[6]), .rowmajor = atoi(f[7]), .name = f[8], .loc = -1};
    } else if (!strcmp(f[0], "v") && nf >= 4) {
      struct var *grown = realloc(o->v, (size_t)(o->nv + 1) * sizeof *grown);
      if (!grown) continue;
      o->v = grown;
      o->v[o->nv++] = (struct var){.size = atoi(f[1]), .type = (GLenum)atoi(f[2]), .name = f[3], .block = -1};
    } else if (!strcmp(f[0], "b") && nf >= 7) {
      /* b binding datasize refvs reffs n idx... name: re-split without a field limit. */
      int n = atoi(f[5]);
      if (n < 0 || nf < 7 + (n > 9 ? 9 : n)) continue;
      struct block *grown = realloc(o->b, (size_t)(o->nb + 1) * sizeof *grown);
      if (!grown) continue;
      o->b = grown;
      struct block *b = &o->b[o->nb];
      *b = (struct block){.binding = atoi(f[1]), .datasize = atoi(f[2]), .refvs = atoi(f[3]), .reffs = atoi(f[4]), .n = n};
      b->indices = calloc((size_t)(n ? n : 1), sizeof *b->indices);
      if (!b->indices) continue;
      /* Fields after f[5] were split only up to 16; walk the rest manually. */
      char *q = f[5] + strlen(f[5]) + 1;
      int i = 0;
      for (; i < n && q < nl; i++) {
        while (*q == ' ') q++;
        b->indices[i] = atoi(q);
        while (*q && *q != ' ') q++;
        if (*q) *q++ = 0;
        else q++;
      }
      while (q < nl && *q == ' ') q++;
      b->name = q < nl ? q : "";
      /* split() cut the line at field 16; rejoin is unnecessary since names have no spaces. */
      o->nb++;
    }
  }
  o->info_valid = 1;
  return 1;
}

static struct obj *program_info(struct glr_ctx *c, GLuint program) {
  struct obj *o = object(c, program);
  if (!o) { add_error(c, GL_INVALID_VALUE); return NULL; }
  if (!o->is_program) { add_error(c, GL_INVALID_OPERATION); return NULL; }
  return load_info(c, program, o) ? o : NULL;
}

static void invalidate(struct glr_ctx *c, GLuint name) {
  struct obj *o = object(c, name);
  if (o) o->info_valid = 0;
}

static void copy_name(const char *name, GLsizei size, GLsizei *length, GLchar *out) {
  GLsizei n = 0;
  if (size > 0 && out) {
    size_t l = strlen(name);
    n = l < (size_t)size - 1 ? (GLsizei)l : size - 1;
    memcpy(out, name, (size_t)n);
    out[n] = 0;
  }
  if (length) *length = n;
}

/* ---- pixel transfer sizes ---- */

static size_t group_bytes(GLenum format, GLenum type) {
  size_t n;
  switch (format) {
  case GL_RED: case GL_RED_INTEGER: case GL_ALPHA: case GL_LUMINANCE: case GL_DEPTH_COMPONENT: n = 1; break;
  case GL_RG: case GL_RG_INTEGER: case GL_LUMINANCE_ALPHA: case GL_DEPTH_STENCIL: n = 2; break;
  case GL_RGB: case GL_RGB_INTEGER: n = 3; break;
  case GL_RGBA: case GL_RGBA_INTEGER: case GL_BGRA_EXT: n = 4; break;
  default: n = 4; break;
  }
  switch (type) {
  case GL_UNSIGNED_BYTE: case GL_BYTE: return n;
  case GL_UNSIGNED_SHORT: case GL_SHORT: case GL_HALF_FLOAT: case GL_HALF_FLOAT_OES: return 2 * n;
  case GL_UNSIGNED_INT: case GL_INT: case GL_FLOAT: return 4 * n;
  case GL_UNSIGNED_SHORT_5_6_5: case GL_UNSIGNED_SHORT_4_4_4_4: case GL_UNSIGNED_SHORT_5_5_5_1: return 2;
  case GL_UNSIGNED_INT_2_10_10_10_REV: case GL_UNSIGNED_INT_10F_11F_11F_REV: case GL_UNSIGNED_INT_5_9_9_9_REV:
  case GL_UNSIGNED_INT_24_8: return 4;
  case GL_FLOAT_32_UNSIGNED_INT_24_8_REV: return 8;
  }
  return 4 * n;
}

/* Bytes from the client pointer an upload/readback touches, skips included. */
static size_t image_bytes(struct glr_ctx *c, int pack, GLsizei w, GLsizei h, GLsizei d, GLenum format, GLenum type, int three_d) {
  if (w <= 0 || h <= 0 || d <= 0) return 0;
  size_t group = group_bytes(format, type);
  GLint align = pack ? c->ps[PS_PACK_ALIGNMENT] : c->ps[PS_UNPACK_ALIGNMENT];
  GLint row_length = pack ? c->ps[PS_PACK_ROW_LENGTH] : c->ps[PS_UNPACK_ROW_LENGTH];
  GLint skip_pixels = pack ? c->ps[PS_PACK_SKIP_PIXELS] : c->ps[PS_UNPACK_SKIP_PIXELS];
  GLint skip_rows = pack ? c->ps[PS_PACK_SKIP_ROWS] : c->ps[PS_UNPACK_SKIP_ROWS];
  GLint image_height = three_d ? c->ps[PS_UNPACK_IMAGE_HEIGHT] : 0;
  GLint skip_images = three_d ? c->ps[PS_UNPACK_SKIP_IMAGES] : 0;
  if (align < 1) align = 1;
  size_t row = (size_t)(row_length > 0 ? row_length : w) * group;
  size_t stride = (row + (size_t)align - 1) / (size_t)align * (size_t)align;
  size_t rows = (size_t)(image_height > 0 ? image_height : h);
  size_t image = stride * rows;
  return (size_t)skip_images * image + (size_t)(d - 1) * image + (size_t)skip_rows * stride + (size_t)(h - 1) * stride +
         (size_t)skip_pixels * group + (size_t)w * group;
}

/* ---- tracked bindings ---- */

static int tex_slot(GLenum target) {
  switch (target) {
  case GL_TEXTURE_2D: return 0;
  case GL_TEXTURE_CUBE_MAP: return 1;
  case GL_TEXTURE_3D: return 2;
  case GL_TEXTURE_2D_ARRAY: return 3;
  }
  return -1;
}
static GLuint *element_slot(struct glr_ctx *c) {
  static _Thread_local GLuint spare;
  if (c->vao > c->next_name[K_VERTEX_ARRAY]) return &spare;
  if (c->vao >= c->nvao) {
    size_t n = c->nvao ? c->nvao : 16;
    while (n <= c->vao) n *= 2;
    GLuint *grown = realloc(c->vao_elements, n * sizeof *grown);
    if (!grown) return &spare;
    memset(grown + c->nvao, 0, (n - c->nvao) * sizeof *grown);
    c->vao_elements = grown;
    c->nvao = n;
  }
  return &c->vao_elements[c->vao];
}
static GLuint *buffer_slot(struct glr_ctx *c, GLenum target) {
  switch (target) {
  case GL_ARRAY_BUFFER: return &c->array_buffer;
  case GL_ELEMENT_ARRAY_BUFFER: return element_slot(c);
  case GL_COPY_READ_BUFFER: return &c->copy_read;
  case GL_COPY_WRITE_BUFFER: return &c->copy_write;
  case GL_PIXEL_PACK_BUFFER: return &c->pack_buffer;
  case GL_PIXEL_UNPACK_BUFFER: return &c->unpack_buffer;
  case GL_UNIFORM_BUFFER: return &c->uniform_buffer;
  case GL_TRANSFORM_FEEDBACK_BUFFER: return &c->tf_buffer;
  }
  return NULL;
}

/* ---- buffer contents (struct gbuf) ---- */

static struct gbuf *gbuf_of(struct glr_ctx *c, GLuint name) {
  if (!name) return NULL;
  if (name >= c->nbufs) {
    size_t n = c->nbufs ? c->nbufs : 64;
    while (n <= name) n *= 2;
    struct gbuf *grown = realloc(c->bufs, n * sizeof *grown);
    if (!grown) return NULL;
    memset(grown + c->nbufs, 0, (n - c->nbufs) * sizeof *grown);
    c->bufs = grown;
    c->nbufs = n;
  }
  return &c->bufs[name];
}
static struct gbuf *bound_gbuf(struct glr_ctx *c, GLenum target) {
  GLuint *slot = buffer_slot(c, target);
  return slot ? gbuf_of(c, *slot) : NULL;
}
static void drop_shadow(struct glr_ctx *c, struct gbuf *g) {
  if (!g || !g->shadow) return;
  free(g->shadow);
  g->shadow = NULL;
  c->shadow_bytes -= g->size;
}
/* The host writes this buffer itself from now on. */
static void host_writes(struct glr_ctx *c, GLuint name) {
  struct gbuf *g = gbuf_of(c, name);
  drop_shadow(c, g);
  if (g) g->host_written = 1;
}
static int valid_usage(GLenum usage) {
  switch (usage) {
  case GL_STREAM_DRAW: case GL_STREAM_READ: case GL_STREAM_COPY:
  case GL_STATIC_DRAW: case GL_STATIC_READ: case GL_STATIC_COPY:
  case GL_DYNAMIC_DRAW: case GL_DYNAMIC_READ: case GL_DYNAMIC_COPY:
    return 1;
  }
  return 0;
}
/* Follows a BufferData sent to the host. */
static void shadow_data(struct glr_ctx *c, GLenum target, size_t size, const void *data, GLenum usage) {
  struct gbuf *g = bound_gbuf(c, target);
  if (!g) return;
  drop_shadow(c, g);
  if (!valid_usage(usage)) return; /* the host rejects it; size unknown */
  g->size = size;
  if (g->host_written || !size || size > SHADOW_MAX || c->shadow_bytes + size > SHADOW_BUDGET) return;
  if (!(g->shadow = malloc(size))) return;
  c->shadow_bytes += size;
  /* The host zero-fills a buffer defined without data. */
  if (data) memcpy(g->shadow, data, size);
  else memset(g->shadow, 0, size);
}
/* Follows a BufferSubData sent to the host. */
static void shadow_sub(struct glr_ctx *c, GLenum target, size_t offset, size_t size, const void *data) {
  struct gbuf *g = bound_gbuf(c, target);
  if (g && g->shadow && data && offset <= g->size && size <= g->size - offset) memcpy(g->shadow + offset, data, size);
}
static int cap_index(GLenum cap) {
  for (unsigned i = 0; i < NCAPS; i++)
    if (caps[i] == cap) return (int)i;
  return -1;
}
/* GL maps buffer objects, not targets: several buffers may be mapped while
 * each is bound to the same target in turn (WebRender's PBO pool does). */
static struct mapping *mapping_of(struct glr_ctx *c, GLenum target) {
  GLuint *slot = buffer_slot(c, target);
  if (!slot || !*slot) return NULL;
  for (int i = 0; i < 8; i++)
    if (c->maps[i].ptr && c->maps[i].buffer == *slot) return &c->maps[i];
  return NULL;
}

/* ---- current context ---- */

static struct glr_ctx *current(void) { return tls_ctx; }
#define CTX struct glr_ctx *c = current(); if (!c) return
#define CTXR(v) struct glr_ctx *c = current(); if (!c) return v

/* ==== OpenGL ES 3.0 ==== */

static void glActiveTexture(GLenum texture) {
  CTX;
  if (texture >= GL_TEXTURE0 && texture - GL_TEXTURE0 < MAX_UNITS) c->unit = texture - GL_TEXTURE0;
  CMD(ActiveTexture, texture);
}
static void glAttachShader(GLuint program, GLuint shader) {
  CTX;
  invalidate(c, program);
  CMD(AttachShader, program, shader);
}
static void glBindAttribLocation(GLuint program, GLuint index, const GLchar *name) {
  CTX;
  if (!name) { add_error(c, GL_INVALID_VALUE); return; }
  uint32_t *w = emit_data(c, OP_BindAttribLocation, 2, B_INLINE, name, strlen(name));
  w[0] = program; w[1] = index;
  done(c);
}
static void glBindBuffer(GLenum target, GLuint buffer) {
  CTX;
  GLuint *slot = buffer_slot(c, target);
  if (slot) *slot = buffer;
  if (target == GL_TRANSFORM_FEEDBACK_BUFFER) host_writes(c, buffer);
  CMD(BindBuffer, target, buffer);
}
static void glBindFramebuffer(GLenum target, GLuint framebuffer) {
  CTX;
  if (target == GL_FRAMEBUFFER || target == GL_DRAW_FRAMEBUFFER) c->draw_fb = framebuffer;
  if (target == GL_FRAMEBUFFER || target == GL_READ_FRAMEBUFFER) c->read_fb = framebuffer;
  CMD(BindFramebuffer, target, framebuffer);
}
static void glBindRenderbuffer(GLenum target, GLuint renderbuffer) {
  CTX;
  if (target == GL_RENDERBUFFER) c->renderbuffer = renderbuffer;
  CMD(BindRenderbuffer, target, renderbuffer);
}
static void glBindTexture(GLenum target, GLuint texture) {
  CTX;
  int slot = tex_slot(target);
  if (slot >= 0) c->tex[c->unit][slot] = texture;
  CMD(BindTexture, target, texture);
}
static void glBlendColor(GLfloat r, GLfloat g, GLfloat b, GLfloat a) { CTX; CMD(BlendColor, fbits(r), fbits(g), fbits(b), fbits(a)); }
static void glBlendEquation(GLenum mode) { CTX; CMD(BlendEquation, mode); }
static void glBlendEquationSeparate(GLenum rgb, GLenum alpha) { CTX; CMD(BlendEquationSeparate, rgb, alpha); }
static void glBlendFunc(GLenum s, GLenum d) { CTX; CMD(BlendFunc, s, d); }
static void glBlendFuncSeparate(GLenum sr, GLenum dr, GLenum sa, GLenum da) { CTX; CMD(BlendFuncSeparate, sr, dr, sa, da); }
static void glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum usage) {
  CTX;
  if (size < 0) { add_error(c, GL_INVALID_VALUE); return; }
  struct mapping *m = mapping_of(c, target);
  if (m) { block_release(c, m->ptr); m->ptr = NULL; }
  uint32_t *w = emit_data(c, OP_BufferData, 2, B_INLINE, data, (size_t)size);
  w[0] = target; w[1] = usage;
  shadow_data(c, target, (size_t)size, data, usage);
  done(c);
}
static void glBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, const void *data) {
  CTX;
  if (size < 0 || offset < 0) { add_error(c, GL_INVALID_VALUE); return; }
  if (!size) return;
  uint32_t *w = emit_data(c, OP_BufferSubData, 2, B_INLINE, data, (size_t)size);
  w[0] = target; w[1] = (uint32_t)offset;
  shadow_sub(c, target, (size_t)offset, (size_t)size, data);
  done(c);
}
static GLenum glCheckFramebufferStatus(GLenum target) {
  CTXR(0);
  int known = target == GL_FRAMEBUFFER || target == GL_DRAW_FRAMEBUFFER || target == GL_READ_FRAMEBUFFER;
  GLuint fb = target == GL_READ_FRAMEBUFFER ? c->read_fb : c->draw_fb;
  if (known && !c->lost && c->fbs_value && c->fbs_gen == c->fb_gen && c->fbs_name == fb) return c->fbs_value;
  double v[1];
  GLenum status = q2(c, OP_CheckFramebufferStatus, target, 0, 1, v, 1) >= 1 ? (GLenum)v[0] : 0;
  if (known && status) {
    c->fbs_gen = c->fb_gen;
    c->fbs_name = fb;
  }
  c->fbs_value = known ? status : 0;
  return status;
}
static void glClear(GLbitfield mask) { CTX; CMD(Clear, mask); }
static void glClearColor(GLfloat r, GLfloat g, GLfloat b, GLfloat a) { CTX; CMD(ClearColor, fbits(r), fbits(g), fbits(b), fbits(a)); }
static void glClearDepthf(GLfloat d) { CTX; CMD(ClearDepthf, fbits(d)); }
static void glClearStencil(GLint s) { CTX; CMD(ClearStencil, (uint32_t)s); }
static void glColorMask(GLboolean r, GLboolean g, GLboolean b, GLboolean a) { CTX; CMD(ColorMask, r, g, b, a); }
static void glCompileShader(GLuint shader) {
  CTX;
  invalidate(c, shader);
  CMD(CompileShader, shader);
}
static int upload_mode(struct glr_ctx *c, const void *data) {
  return c->unpack_buffer ? B_OFFSET : data ? B_INLINE : B_NULL;
}
static void glCompressedTexImage2D(GLenum target, GLint level, GLenum ifmt, GLsizei w, GLsizei h, GLint border, GLsizei size, const void *data) {
  CTX;
  uint32_t *o = emit_data(c, OP_CompressedTexImage2D, 6, upload_mode(c, data), data, (size_t)(size > 0 ? size : 0));
  o[0] = target; o[1] = (uint32_t)level; o[2] = ifmt; o[3] = (uint32_t)w; o[4] = (uint32_t)h; o[5] = (uint32_t)border;
  done(c);
}
static void glCompressedTexSubImage2D(GLenum target, GLint level, GLint x, GLint y, GLsizei w, GLsizei h, GLenum format, GLsizei size, const void *data) {
  CTX;
  uint32_t *o = emit_data(c, OP_CompressedTexSubImage2D, 7, upload_mode(c, data), data, (size_t)(size > 0 ? size : 0));
  o[0] = target; o[1] = (uint32_t)level; o[2] = (uint32_t)x; o[3] = (uint32_t)y; o[4] = (uint32_t)w; o[5] = (uint32_t)h; o[6] = format;
  done(c);
}
static void glCopyTexImage2D(GLenum target, GLint level, GLenum ifmt, GLint x, GLint y, GLsizei w, GLsizei h, GLint border) {
  CTX;
  CMD(CopyTexImage2D, target, (uint32_t)level, ifmt, (uint32_t)x, (uint32_t)y, (uint32_t)w, (uint32_t)h, (uint32_t)border);
}
static void glCopyTexSubImage2D(GLenum target, GLint level, GLint xo, GLint yo, GLint x, GLint y, GLsizei w, GLsizei h) {
  CTX;
  CMD(CopyTexSubImage2D, target, (uint32_t)level, (uint32_t)xo, (uint32_t)yo, (uint32_t)x, (uint32_t)y, (uint32_t)w, (uint32_t)h);
}
static GLuint glCreateProgram(void) {
  CTXR(0);
  GLuint name = ++c->next_name[K_PROGRAM];
  if (!new_object(c, name, 1, 0)) { add_error(c, GL_OUT_OF_MEMORY); return 0; }
  CMD(CreateProgram, name);
  return name;
}
static GLuint glCreateShader(GLenum type) {
  CTXR(0);
  if (type != GL_VERTEX_SHADER && type != GL_FRAGMENT_SHADER) { add_error(c, GL_INVALID_ENUM); return 0; }
  GLuint name = ++c->next_name[K_PROGRAM];
  if (!new_object(c, name, 0, type)) { add_error(c, GL_OUT_OF_MEMORY); return 0; }
  CMD(CreateShader, type, name);
  return name;
}
static void glCullFace(GLenum mode) { CTX; CMD(CullFace, mode); }

static void delete_names(struct glr_ctx *c, int kind, GLsizei n, const GLuint *names) {
  if (n < 0) { add_error(c, GL_INVALID_VALUE); return; }
  if (!n || !names) return;
  for (GLsizei i = 0; i < n; i++) {
    GLuint x = names[i];
    if (!x) continue;
    switch (kind) {
    case K_BUFFER:
      if (c->array_buffer == x) c->array_buffer = 0;
      if (c->copy_read == x) c->copy_read = 0;
      if (c->copy_write == x) c->copy_write = 0;
      if (c->pack_buffer == x) c->pack_buffer = 0;
      if (c->unpack_buffer == x) c->unpack_buffer = 0;
      if (*element_slot(c) == x) *element_slot(c) = 0;
      if (c->uniform_buffer == x) c->uniform_buffer = 0;
      if (c->tf_buffer == x) c->tf_buffer = 0;
      for (int k = 0; k < 8; k++)
        if (c->maps[k].ptr && c->maps[k].buffer == x) { free(c->maps[k].ptr); c->maps[k].ptr = NULL; }
      if (x < c->nbufs) {
        drop_shadow(c, &c->bufs[x]);
        c->bufs[x] = (struct gbuf){0};
      }
      break;
    case K_TEXTURE:
      for (unsigned u = 0; u < MAX_UNITS; u++)
        for (int t = 0; t < 4; t++)
          if (c->tex[u][t] == x) c->tex[u][t] = 0;
      break;
    case K_SAMPLER:
      for (unsigned u = 0; u < MAX_UNITS; u++)
        if (c->sampler[u] == x) c->sampler[u] = 0;
      break;
    case K_FRAMEBUFFER:
      if (c->draw_fb == x) c->draw_fb = 0;
      if (c->read_fb == x) c->read_fb = 0;
      break;
    case K_RENDERBUFFER:
      if (c->renderbuffer == x) c->renderbuffer = 0;
      break;
    case K_VERTEX_ARRAY:
      if (c->vao == x) c->vao = 0;
      if (x < c->nvao) c->vao_elements[x] = 0;
      break;
    case K_TRANSFORM_FEEDBACK:
      if (c->tfo == x) c->tfo = 0;
      break;
    }
  }
  uint32_t *w = emit(c, OP_DeleteObjects, 2 + (uint32_t)n);
  w[0] = (uint32_t)kind;
  w[1] = (uint32_t)n;
  memcpy(w + 2, names, (size_t)n * 4);
}
static void glDeleteBuffers(GLsizei n, const GLuint *b) { CTX; delete_names(c, K_BUFFER, n, b); }
static void glDeleteFramebuffers(GLsizei n, const GLuint *b) { CTX; delete_names(c, K_FRAMEBUFFER, n, b); }
static void glDeleteRenderbuffers(GLsizei n, const GLuint *b) { CTX; delete_names(c, K_RENDERBUFFER, n, b); }
static void glDeleteTextures(GLsizei n, const GLuint *b) { CTX; delete_names(c, K_TEXTURE, n, b); }
static void glDeleteQueries(GLsizei n, const GLuint *b) { CTX; delete_names(c, K_QUERY, n, b); }
static void glDeleteSamplers(GLsizei n, const GLuint *b) { CTX; delete_names(c, K_SAMPLER, n, b); }
static void glDeleteVertexArrays(GLsizei n, const GLuint *b) { CTX; delete_names(c, K_VERTEX_ARRAY, n, b); }
static void glDeleteTransformFeedbacks(GLsizei n, const GLuint *b) { CTX; delete_names(c, K_TRANSFORM_FEEDBACK, n, b); }
static void glDeleteProgram(GLuint program) {
  CTX;
  if (!program) return;
  invalidate(c, program);
  CMD(DeleteProgram, program);
}
static void glDeleteShader(GLuint shader) {
  CTX;
  if (!shader) return;
  invalidate(c, shader);
  CMD(DeleteShader, shader);
}
static void glDepthFunc(GLenum f) { CTX; CMD(DepthFunc, f); }
static void glDepthMask(GLboolean flag) { CTX; CMD(DepthMask, flag); }
static void glDepthRangef(GLfloat n, GLfloat f) { CTX; CMD(DepthRangef, fbits(n), fbits(f)); }
static void glDetachShader(GLuint program, GLuint shader) {
  CTX;
  invalidate(c, program);
  CMD(DetachShader, program, shader);
}
static void set_cap(struct glr_ctx *c, GLenum cap, int on) {
  int i = cap_index(cap);
  if (i >= 0) c->enabled[i] = (unsigned char)on;
}
static void glDisable(GLenum cap) { CTX; set_cap(c, cap, 0); CMD(Disable, cap); }
static void glEnable(GLenum cap) { CTX; set_cap(c, cap, 1); CMD(Enable, cap); }
static void glDisableVertexAttribArray(GLuint i) { CTX; CMD(DisableVertexAttribArray, i); }
static void glEnableVertexAttribArray(GLuint i) { CTX; CMD(EnableVertexAttribArray, i); }
static void glDrawArrays(GLenum mode, GLint first, GLsizei count) { CTX; CMD(DrawArrays, mode, (uint32_t)first, (uint32_t)count); }

static size_t index_bytes(GLenum type) {
  return type == GL_UNSIGNED_INT ? 4 : type == GL_UNSIGNED_SHORT ? 2 : 1;
}
/* Client-memory indices (no element buffer bound) go through a scratch buffer. */
static uint32_t index_offset(struct glr_ctx *c, GLsizei count, GLenum type, const void *indices, int *scratch) {
  *scratch = 0;
  if (*element_slot(c) || !indices || count <= 0) return (uint32_t)(uintptr_t)indices;
  if (!c->scratch_elements) c->scratch_elements = ++c->next_name[K_BUFFER];
  *scratch = 1;
  CMD(BindBuffer, GL_ELEMENT_ARRAY_BUFFER, c->scratch_elements);
  uint32_t *w = emit_data(c, OP_BufferData, 2, B_INLINE, indices, (size_t)count * index_bytes(type));
  w[0] = GL_ELEMENT_ARRAY_BUFFER;
  w[1] = GL_STREAM_DRAW;
  return 0;
}
static void index_restore(struct glr_ctx *c, int scratch) {
  if (scratch) CMD(BindBuffer, GL_ELEMENT_ARRAY_BUFFER, 0);
  done(c);
}
static void glDrawElements(GLenum mode, GLsizei count, GLenum type, const void *indices) {
  CTX;
  int scratch;
  uint32_t offset = index_offset(c, count, type, indices, &scratch);
  CMD(DrawElements, mode, (uint32_t)count, type, offset);
  index_restore(c, scratch);
}
static void glFinish(void) {
  CTX;
  CMD0(Finish);
  flush(c);
}
static void glFlush(void) {
  CTX;
  CMD0(Flush);
  flush(c);
}
static void glFramebufferRenderbuffer(GLenum target, GLenum attachment, GLenum rbtarget, GLuint rb) {
  CTX;
  CMD(FramebufferRenderbuffer, target, attachment, rbtarget, rb);
}
static void glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
  CTX;
  CMD(FramebufferTexture2D, target, attachment, textarget, texture, (uint32_t)level);
}
static void glFrontFace(GLenum mode) { CTX; CMD(FrontFace, mode); }
static void glGenBuffers(GLsizei n, GLuint *b) { CTX; gen_names(c, K_BUFFER, n, b); }
static void glGenFramebuffers(GLsizei n, GLuint *b) { CTX; gen_names(c, K_FRAMEBUFFER, n, b); }
static void glGenRenderbuffers(GLsizei n, GLuint *b) { CTX; gen_names(c, K_RENDERBUFFER, n, b); }
static void glGenTextures(GLsizei n, GLuint *b) { CTX; gen_names(c, K_TEXTURE, n, b); }
static void glGenQueries(GLsizei n, GLuint *b) { CTX; gen_names(c, K_QUERY, n, b); }
static void glGenSamplers(GLsizei n, GLuint *b) { CTX; gen_names(c, K_SAMPLER, n, b); }
static void glGenVertexArrays(GLsizei n, GLuint *b) { CTX; gen_names(c, K_VERTEX_ARRAY, n, b); }
static void glGenTransformFeedbacks(GLsizei n, GLuint *b) { CTX; gen_names(c, K_TRANSFORM_FEEDBACK, n, b); }
static void glGenerateMipmap(GLenum target) { CTX; CMD(GenerateMipmap, target); }

static void glGetActiveAttrib(GLuint program, GLuint index, GLsizei size, GLsizei *length, GLint *vsize, GLenum *type, GLchar *name) {
  CTX;
  struct obj *o = program_info(c, program);
  if (!o) return;
  if (index >= (GLuint)o->na) { add_error(c, GL_INVALID_VALUE); return; }
  copy_name(o->a[index].name, size, length, name);
  if (vsize) *vsize = o->a[index].size;
  if (type) *type = o->a[index].type;
}
static void glGetActiveUniform(GLuint program, GLuint index, GLsizei size, GLsizei *length, GLint *vsize, GLenum *type, GLchar *name) {
  CTX;
  struct obj *o = program_info(c, program);
  if (!o) return;
  if (index >= (GLuint)o->nu) { add_error(c, GL_INVALID_VALUE); return; }
  copy_name(o->u[index].name, size, length, name);
  if (vsize) *vsize = o->u[index].size;
  if (type) *type = o->u[index].type;
}
static void glGetAttachedShaders(GLuint program, GLsizei max, GLsizei *count, GLuint *shaders) {
  CTX;
  struct obj *o = program_info(c, program);
  GLsizei n = 0;
  if (o)
    for (; n < o->nattached && n < max; n++) shaders[n] = o->attached[n];
  if (count) *count = n;
}
static GLint glGetAttribLocation(GLuint program, const GLchar *name) {
  CTXR(-1);
  struct obj *o = program_info(c, program);
  if (!o || !name) return -1;
  if (!o->status) { add_error(c, GL_INVALID_OPERATION); return -1; }
  for (int i = 0; i < o->na; i++)
    if (!strcmp(o->a[i].name, name)) return o->a[i].loc;
  return -1;
}

/* Generic getters: state answered here, then the host. */
static const char *const *ext_list(struct glr_ctx *c) { return c->ext; }
static int local_get(struct glr_ctx *c, GLenum pname, double *v) {
  switch (pname) {
  case GL_MAJOR_VERSION: v[0] = 3; return 1;
  case GL_MINOR_VERSION: v[0] = 0; return 1;
  case GL_NUM_EXTENSIONS: v[0] = c->n_ext; return 1;
  case GL_NUM_SHADER_BINARY_FORMATS: case GL_NUM_PROGRAM_BINARY_FORMATS: v[0] = 0; return 1;
  case GL_SHADER_BINARY_FORMATS: case GL_PROGRAM_BINARY_FORMATS: return 0;
  case GL_SHADER_COMPILER: v[0] = 1; return 1;
  case GL_ACTIVE_TEXTURE: v[0] = GL_TEXTURE0 + c->unit; return 1;
  case GL_TEXTURE_BINDING_2D: v[0] = c->tex[c->unit][0]; return 1;
  case GL_TEXTURE_BINDING_CUBE_MAP: v[0] = c->tex[c->unit][1]; return 1;
  case GL_TEXTURE_BINDING_3D: v[0] = c->tex[c->unit][2]; return 1;
  case GL_TEXTURE_BINDING_2D_ARRAY: v[0] = c->tex[c->unit][3]; return 1;
  case GL_SAMPLER_BINDING: v[0] = c->sampler[c->unit]; return 1;
  case GL_ARRAY_BUFFER_BINDING: v[0] = c->array_buffer; return 1;
  case GL_ELEMENT_ARRAY_BUFFER_BINDING: v[0] = *element_slot(c); return 1;
  case GL_COPY_READ_BUFFER_BINDING: v[0] = c->copy_read; return 1;
  case GL_COPY_WRITE_BUFFER_BINDING: v[0] = c->copy_write; return 1;
  case GL_PIXEL_PACK_BUFFER_BINDING: v[0] = c->pack_buffer; return 1;
  case GL_PIXEL_UNPACK_BUFFER_BINDING: v[0] = c->unpack_buffer; return 1;
  case GL_DRAW_FRAMEBUFFER_BINDING: v[0] = c->draw_fb; return 1;
  case GL_READ_FRAMEBUFFER_BINDING: v[0] = c->read_fb; return 1;
  case GL_RENDERBUFFER_BINDING: v[0] = c->renderbuffer; return 1;
  case GL_CURRENT_PROGRAM: v[0] = c->program; return 1;
  case GL_VERTEX_ARRAY_BINDING: v[0] = c->vao; return 1;
  case GL_TRANSFORM_FEEDBACK_BINDING: v[0] = c->tfo; return 1;
  case GL_VIEWPORT: for (int i = 0; i < 4; i++) v[i] = c->viewport[i]; return 4;
  case GL_SCISSOR_BOX: for (int i = 0; i < 4; i++) v[i] = c->scissor[i]; return 4;
  }
  for (int i = 0; i < PS_COUNT; i++)
    if (ps_names[i] == pname) { v[0] = c->ps[i]; return 1; }
  int cap = cap_index(pname);
  if (cap >= 0) { v[0] = c->enabled[cap]; return 1; }
  for (int i = 0; i < c->nlimits; i++)
    if (c->limits[i].pname == pname) {
      int n = c->limits[i].n < MAX_VALUES ? c->limits[i].n : MAX_VALUES;
      memcpy(v, c->limits[i].v, (size_t)n * sizeof *v);
      return n;
    }
  return -1;
}
static int get_values(struct glr_ctx *c, GLenum pname, double *v) {
  int n = local_get(c, pname, v);
  if (n >= 0) return n;
  n = q2(c, OP_GetParameter, pname, 0, 1, v, MAX_VALUES);
  return n < 0 ? 0 : n < MAX_VALUES ? n : MAX_VALUES;
}
static int normalized_pname(GLenum pname) {
  return pname == GL_COLOR_CLEAR_VALUE || pname == GL_BLEND_COLOR || pname == GL_DEPTH_RANGE || pname == GL_DEPTH_CLEAR_VALUE ||
         pname == GL_SAMPLE_COVERAGE_VALUE;
}
static GLint to_int(double v, int normalized) {
  if (normalized) v = v * 2147483647.0;
  if (v >= 2147483647.0) return 2147483647;
  if (v <= -2147483648.0) return (GLint)-2147483647 - 1;
  return (GLint)llround(v);
}
static void glGetBooleanv(GLenum pname, GLboolean *data) {
  CTX;
  double v[MAX_VALUES];
  int n = get_values(c, pname, v);
  for (int i = 0; i < n; i++) data[i] = v[i] != 0;
}
static void glGetFloatv(GLenum pname, GLfloat *data) {
  CTX;
  double v[MAX_VALUES];
  int n = get_values(c, pname, v);
  for (int i = 0; i < n; i++) data[i] = (GLfloat)v[i];
}
static void glGetIntegerv(GLenum pname, GLint *data) {
  CTX;
  double v[MAX_VALUES];
  int n = get_values(c, pname, v);
  for (int i = 0; i < n; i++) data[i] = to_int(v[i], normalized_pname(pname));
}
static void glGetInteger64v(GLenum pname, GLint64 *data) {
  CTX;
  double v[MAX_VALUES];
  int n = get_values(c, pname, v);
  for (int i = 0; i < n; i++) data[i] = normalized_pname(pname) ? (GLint64)to_int(v[i], 1) : (GLint64)llround(v[i]);
}
static void glGetIntegeri_v(GLenum target, GLuint index, GLint *data) {
  CTX;
  double v[4];
  if (q2(c, OP_GetIndexedParameter, target, index, 2, v, 1) >= 1) data[0] = to_int(v[0], 0);
}
static void glGetInteger64i_v(GLenum target, GLuint index, GLint64 *data) {
  CTX;
  double v[4];
  if (q2(c, OP_GetIndexedParameter, target, index, 2, v, 1) >= 1) data[0] = (GLint64)llround(v[0]);
}
static int buffer_param_local(struct glr_ctx *c, GLenum target, GLenum pname, double *v) {
  struct mapping *m = mapping_of(c, target);
  switch (pname) {
  case GL_BUFFER_MAPPED: v[0] = m != NULL; return 1;
  case GL_BUFFER_ACCESS_FLAGS: v[0] = m ? m->access : 0; return 1;
  case GL_BUFFER_MAP_LENGTH: v[0] = m ? (double)m->length : 0; return 1;
  case GL_BUFFER_MAP_OFFSET: v[0] = m ? (double)m->offset : 0; return 1;
  }
  return q2(c, OP_GetBufferParameter, target, pname, 2, v, 1);
}
static void glGetBufferParameteriv(GLenum target, GLenum pname, GLint *params) {
  CTX;
  double v[1];
  if (buffer_param_local(c, target, pname, v) >= 1) params[0] = to_int(v[0], 0);
}
static void glGetBufferParameteri64v(GLenum target, GLenum pname, GLint64 *params) {
  CTX;
  double v[1];
  if (buffer_param_local(c, target, pname, v) >= 1) params[0] = (GLint64)llround(v[0]);
}
static void glGetBufferPointerv(GLenum target, GLenum pname, void **params) {
  CTX;
  struct mapping *m = mapping_of(c, target);
  if (pname != GL_BUFFER_MAP_POINTER) { add_error(c, GL_INVALID_ENUM); return; }
  *params = m ? m->ptr : NULL;
}
static GLenum glGetError(void) {
  CTXR(GL_NO_ERROR);
  if (c->error) {
    GLenum e = c->error;
    c->error = 0;
    return e;
  }
  if (c->lost || (!c->since_error && !c->used)) return GL_NO_ERROR;
  double v[1];
  GLenum e = q2(c, OP_GetError, 0, 0, 0, v, 1) >= 1 ? (GLenum)v[0] : GL_NO_ERROR;
  if (e == GL_NO_ERROR) c->since_error = 0;
  return e;
}
static void glGetFramebufferAttachmentParameteriv(GLenum target, GLenum attachment, GLenum pname, GLint *params) {
  CTX;
  double v[1];
  if (q3(c, OP_GetFramebufferAttachmentParameter, target, attachment, pname, v, 1) >= 1) params[0] = to_int(v[0], 0);
}
static void glGetProgramiv(GLuint program, GLenum pname, GLint *params) {
  CTX;
  struct obj *o = program_info(c, program);
  if (!o) return;
  GLint v = 0, max = 0;
  switch (pname) {
  case GL_DELETE_STATUS: v = o->deleted; break;
  case GL_LINK_STATUS: v = o->status; break;
  case GL_VALIDATE_STATUS: v = o->validate; break;
  case GL_INFO_LOG_LENGTH: v = o->log_len ? o->log_len + 1 : 0; break;
  case GL_ATTACHED_SHADERS: v = o->nattached; break;
  case GL_ACTIVE_ATTRIBUTES: v = o->na; break;
  case GL_ACTIVE_ATTRIBUTE_MAX_LENGTH:
    for (int i = 0; i < o->na; i++) if ((GLint)strlen(o->a[i].name) + 1 > max) max = (GLint)strlen(o->a[i].name) + 1;
    v = max; break;
  case GL_ACTIVE_UNIFORMS: v = o->nu; break;
  case GL_ACTIVE_UNIFORM_MAX_LENGTH:
    for (int i = 0; i < o->nu; i++) if ((GLint)strlen(o->u[i].name) + 1 > max) max = (GLint)strlen(o->u[i].name) + 1;
    v = max; break;
  case GL_ACTIVE_UNIFORM_BLOCKS: v = o->nb; break;
  case GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH:
    for (int i = 0; i < o->nb; i++) if ((GLint)strlen(o->b[i].name) + 1 > max) max = (GLint)strlen(o->b[i].name) + 1;
    v = max; break;
  case GL_TRANSFORM_FEEDBACK_VARYINGS: v = o->nv; break;
  case GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH:
    for (int i = 0; i < o->nv; i++) if ((GLint)strlen(o->v[i].name) + 1 > max) max = (GLint)strlen(o->v[i].name) + 1;
    v = max; break;
  case GL_TRANSFORM_FEEDBACK_BUFFER_MODE: v = o->tfmode; break;
  case GL_PROGRAM_BINARY_RETRIEVABLE_HINT: case GL_PROGRAM_BINARY_LENGTH: v = 0; break;
  default: add_error(c, GL_INVALID_ENUM); return;
  }
  params[0] = v;
}
static void glGetProgramInfoLog(GLuint program, GLsizei size, GLsizei *length, GLchar *log) {
  CTX;
  struct obj *o = program_info(c, program);
  copy_name(o && o->log ? o->log : "", size, length, log);
}
static void glGetRenderbufferParameteriv(GLenum target, GLenum pname, GLint *params) {
  CTX;
  double v[1];
  if (q2(c, OP_GetRenderbufferParameter, target, pname, 2, v, 1) >= 1) params[0] = to_int(v[0], 0);
}
static struct obj *shader_info(struct glr_ctx *c, GLuint shader) {
  struct obj *o = object(c, shader);
  if (!o) { add_error(c, GL_INVALID_VALUE); return NULL; }
  if (o->is_program) { add_error(c, GL_INVALID_OPERATION); return NULL; }
  return o;
}
static void glGetShaderiv(GLuint shader, GLenum pname, GLint *params) {
  CTX;
  struct obj *o = shader_info(c, shader);
  if (!o) return;
  switch (pname) {
  case GL_SHADER_TYPE: params[0] = (GLint)o->type; return;
  case GL_SHADER_SOURCE_LENGTH: params[0] = o->source ? (GLint)strlen(o->source) + 1 : 0; return;
  case GL_DELETE_STATUS: case GL_COMPILE_STATUS: case GL_INFO_LOG_LENGTH:
    if (!load_info(c, shader, o)) return;
    params[0] = pname == GL_DELETE_STATUS ? o->deleted : pname == GL_COMPILE_STATUS ? o->status : o->log_len ? o->log_len + 1 : 0;
    return;
  }
  add_error(c, GL_INVALID_ENUM);
}
static void glGetShaderInfoLog(GLuint shader, GLsizei size, GLsizei *length, GLchar *log) {
  CTX;
  struct obj *o = shader_info(c, shader);
  copy_name(o && load_info(c, shader, o) && o->log ? o->log : "", size, length, log);
}
static void glGetShaderPrecisionFormat(GLenum shadertype, GLenum precisiontype, GLint *range, GLint *precision) {
  CTX;
  double v[3] = {0, 0, 0};
  if (q2(c, OP_GetShaderPrecisionFormat, shadertype, precisiontype, 2, v, 3) >= 3) {
    range[0] = (GLint)v[0];
    range[1] = (GLint)v[1];
    *precision = (GLint)v[2];
  }
}
static void glGetShaderSource(GLuint shader, GLsizei size, GLsizei *length, GLchar *source) {
  CTX;
  struct obj *o = shader_info(c, shader);
  copy_name(o && o->source ? o->source : "", size, length, source);
}
static const GLubyte *glGetString(GLenum name) {
  CTXR(NULL);
  switch (name) {
  case GL_VENDOR: return (const GLubyte *)c->vendor;
  case GL_RENDERER: return (const GLubyte *)c->renderer;
  case GL_VERSION: return (const GLubyte *)c->version;
  case GL_SHADING_LANGUAGE_VERSION: return (const GLubyte *)c->glsl;
  case GL_EXTENSIONS: return (const GLubyte *)c->extensions;
  }
  add_error(c, GL_INVALID_ENUM);
  return NULL;
}
static const GLubyte *glGetStringi(GLenum name, GLuint index) {
  CTXR(NULL);
  if (name != GL_EXTENSIONS) { add_error(c, GL_INVALID_ENUM); return NULL; }
  if (index >= (GLuint)c->n_ext) { add_error(c, GL_INVALID_VALUE); return NULL; }
  return (const GLubyte *)ext_list(c)[index];
}
static void glGetTexParameterfv(GLenum target, GLenum pname, GLfloat *params) {
  CTX;
  double v[1];
  if (q2(c, OP_GetTexParameter, target, pname, 2, v, 1) >= 1) params[0] = (GLfloat)v[0];
}
static void glGetTexParameteriv(GLenum target, GLenum pname, GLint *params) {
  CTX;
  double v[1];
  if (q2(c, OP_GetTexParameter, target, pname, 2, v, 1) >= 1) params[0] = to_int(v[0], 0);
}
static int get_uniform(struct glr_ctx *c, GLuint program, GLint location, double *v) {
  if (location < 0) { add_error(c, GL_INVALID_OPERATION); return 0; }
  int n = q2(c, OP_GetUniform, program, (uint32_t)location, 2, v, 16);
  return n < 0 ? 0 : n > 16 ? 16 : n;
}
static void glGetUniformfv(GLuint program, GLint location, GLfloat *params) {
  CTX;
  double v[16];
  int n = get_uniform(c, program, location, v);
  for (int i = 0; i < n; i++) params[i] = (GLfloat)v[i];
}
static void glGetUniformiv(GLuint program, GLint location, GLint *params) {
  CTX;
  double v[16];
  int n = get_uniform(c, program, location, v);
  for (int i = 0; i < n; i++) params[i] = (GLint)v[i];
}
static void glGetUniformuiv(GLuint program, GLint location, GLuint *params) {
  CTX;
  double v[16];
  int n = get_uniform(c, program, location, v);
  for (int i = 0; i < n; i++) params[i] = (GLuint)v[i];
}
/* Locations are (active uniform index << 16) | array element. */
static GLint glGetUniformLocation(GLuint program, const GLchar *name) {
  CTXR(-1);
  struct obj *o = program_info(c, program);
  if (!o || !name) return -1;
  if (!o->status) { add_error(c, GL_INVALID_OPERATION); return -1; }
  size_t len = strlen(name);
  for (int i = 0; i < o->nu && i < 32768; i++) {
    const struct var *u = &o->u[i];
    if (u->block >= 0) continue;
    size_t base = strlen(u->name);
    if (base >= 3 && !strcmp(u->name + base - 3, "[0]")) base -= 3;
    if (!strcmp(name, u->name)) return i << 16;
    if (len < base || strncmp(name, u->name, base)) continue;
    if (len == base) return i << 16;
    if (name[base] != '[') continue;
    char *end;
    unsigned long element = strtoul(name + base + 1, &end, 10);
    if (end == name + base + 1 || strcmp(end, "]") || element >= (unsigned long)u->size || element > 0xffff) continue;
    return (GLint)((i << 16) | (int)element);
  }
  return -1;
}
static void get_vertex_attrib(struct glr_ctx *c, GLuint index, GLenum pname, double *v, int *n) {
  *n = q2(c, OP_GetVertexAttrib, index, pname, 2, v, 4);
  if (*n > 4) *n = 4;
}
static void glGetVertexAttribfv(GLuint index, GLenum pname, GLfloat *params) {
  CTX;
  double v[4];
  int n;
  get_vertex_attrib(c, index, pname, v, &n);
  for (int i = 0; i < n; i++) params[i] = (GLfloat)v[i];
}
static void glGetVertexAttribiv(GLuint index, GLenum pname, GLint *params) {
  CTX;
  double v[4];
  int n;
  get_vertex_attrib(c, index, pname, v, &n);
  for (int i = 0; i < n; i++) params[i] = to_int(v[i], 0);
}
static void glGetVertexAttribIiv(GLuint index, GLenum pname, GLint *params) { glGetVertexAttribiv(index, pname, params); }
static void glGetVertexAttribIuiv(GLuint index, GLenum pname, GLuint *params) {
  CTX;
  double v[4];
  int n;
  get_vertex_attrib(c, index, pname, v, &n);
  for (int i = 0; i < n; i++) params[i] = (GLuint)v[i];
}
static void glGetVertexAttribPointerv(GLuint index, GLenum pname, void **pointer) {
  CTX;
  double v[1];
  if (q2(c, OP_GetVertexAttribOffset, index, pname, 2, v, 1) >= 1) *pointer = (void *)(uintptr_t)v[0];
}
static void glHint(GLenum target, GLenum mode) { CTX; CMD(Hint, target, mode); }
static GLboolean is_object(int kind, GLuint name) {
  CTXR(GL_FALSE);
  if (!name) return GL_FALSE;
  double v[1];
  return q2(c, OP_IsObject, (uint32_t)kind, name, 2, v, 1) >= 1 && v[0] != 0;
}
static GLboolean glIsBuffer(GLuint b) { return is_object(K_BUFFER, b); }
static GLboolean glIsFramebuffer(GLuint b) { return is_object(K_FRAMEBUFFER, b); }
static GLboolean glIsRenderbuffer(GLuint b) { return is_object(K_RENDERBUFFER, b); }
static GLboolean glIsTexture(GLuint b) { return is_object(K_TEXTURE, b); }
static GLboolean glIsQuery(GLuint b) { return is_object(K_QUERY, b); }
static GLboolean glIsSampler(GLuint b) { return is_object(K_SAMPLER, b); }
static GLboolean glIsVertexArray(GLuint b) { return is_object(K_VERTEX_ARRAY, b); }
static GLboolean glIsTransformFeedback(GLuint b) { return is_object(K_TRANSFORM_FEEDBACK, b); }
static GLboolean glIsProgram(GLuint b) { return is_object(K_PROGRAM, b); }
static GLboolean glIsShader(GLuint b) { return is_object(K_SHADER, b); }
static GLboolean glIsEnabled(GLenum cap) {
  CTXR(GL_FALSE);
  int i = cap_index(cap);
  if (i >= 0) return c->enabled[i];
  double v[1];
  return q2(c, OP_IsEnabled, cap, 0, 1, v, 1) >= 1 && v[0] != 0;
}
static void glLineWidth(GLfloat width) { CTX; CMD(LineWidth, fbits(width)); }
static void glLinkProgram(GLuint program) {
  CTX;
  invalidate(c, program);
  CMD(LinkProgram, program);
}
static void glPixelStorei(GLenum pname, GLint param) {
  CTX;
  for (int i = 0; i < PS_COUNT; i++)
    if (ps_names[i] == pname) c->ps[i] = param;
  CMD(PixelStorei, pname, (uint32_t)param);
}
static void glPolygonOffset(GLfloat factor, GLfloat units) { CTX; CMD(PolygonOffset, fbits(factor), fbits(units)); }
static void glReadPixels(GLint x, GLint y, GLsizei w, GLsizei h, GLenum format, GLenum type, void *pixels) {
  CTX;
  if (w < 0 || h < 0) { add_error(c, GL_INVALID_VALUE); return; }
  size_t bytes = image_bytes(c, 1, w, h, 1, format, type, 0);
  int mode = c->pack_buffer ? B_OFFSET : B_POINTER;
  if (c->pack_buffer) drop_shadow(c, gbuf_of(c, c->pack_buffer));
  if (mode == B_POINTER && (!pixels || !bytes)) {
    if (!pixels && bytes) add_error(c, GL_INVALID_OPERATION);
    return;
  }
  uint32_t *o = emit_data(c, OP_ReadPixels, 6, mode, pixels, bytes);
  o[0] = (uint32_t)x; o[1] = (uint32_t)y; o[2] = (uint32_t)w; o[3] = (uint32_t)h; o[4] = format; o[5] = type;
  done(c);
}
static void glReleaseShaderCompiler(void) {}
static void glRenderbufferStorage(GLenum target, GLenum ifmt, GLsizei w, GLsizei h) {
  CTX;
  CMD(RenderbufferStorage, target, ifmt, (uint32_t)w, (uint32_t)h);
}
static void glSampleCoverage(GLfloat value, GLboolean invert) { CTX; CMD(SampleCoverage, fbits(value), invert); }
static void glScissor(GLint x, GLint y, GLsizei w, GLsizei h) {
  CTX;
  c->scissor[0] = x; c->scissor[1] = y; c->scissor[2] = w; c->scissor[3] = h;
  CMD(Scissor, (uint32_t)x, (uint32_t)y, (uint32_t)w, (uint32_t)h);
}
static void glShaderBinary(GLsizei count, const GLuint *shaders, GLenum format, const void *binary, GLsizei length) {
  (void)count; (void)shaders; (void)format; (void)binary; (void)length;
  CTX;
  add_error(c, GL_INVALID_ENUM); /* no binary formats */
}
static void glShaderSource(GLuint shader, GLsizei count, const GLchar *const *strings, const GLint *lengths) {
  CTX;
  struct obj *o = shader_info(c, shader);
  if (!o) return;
  if (count < 0) { add_error(c, GL_INVALID_VALUE); return; }
  size_t total = 0;
  for (GLsizei i = 0; i < count; i++)
    total += strings[i] ? (lengths && lengths[i] >= 0 ? (size_t)lengths[i] : strlen(strings[i])) : 0;
  char *source = malloc(total + 1);
  if (!source) { add_error(c, GL_OUT_OF_MEMORY); return; }
  size_t used = 0;
  for (GLsizei i = 0; i < count; i++) {
    if (!strings[i]) continue;
    size_t l = lengths && lengths[i] >= 0 ? (size_t)lengths[i] : strlen(strings[i]);
    memcpy(source + used, strings[i], l);
    used += l;
  }
  source[used] = 0;
  free(o->source);
  o->source = source;
  o->info_valid = 0;
  uint32_t *w = emit_data(c, OP_ShaderSource, 1, B_INLINE, source, used);
  w[0] = shader;
  done(c);
}
static void glStencilFunc(GLenum f, GLint ref, GLuint mask) { CTX; CMD(StencilFunc, f, (uint32_t)ref, mask); }
static void glStencilFuncSeparate(GLenum face, GLenum f, GLint ref, GLuint mask) { CTX; CMD(StencilFuncSeparate, face, f, (uint32_t)ref, mask); }
static void glStencilMask(GLuint mask) { CTX; CMD(StencilMask, mask); }
static void glStencilMaskSeparate(GLenum face, GLuint mask) { CTX; CMD(StencilMaskSeparate, face, mask); }
static void glStencilOp(GLenum a, GLenum b, GLenum d) { CTX; CMD(StencilOp, a, b, d); }
static void glStencilOpSeparate(GLenum face, GLenum a, GLenum b, GLenum d) { CTX; CMD(StencilOpSeparate, face, a, b, d); }
static void glTexImage2D(GLenum target, GLint level, GLint ifmt, GLsizei w, GLsizei h, GLint border, GLenum format, GLenum type, const void *pixels) {
  CTX;
  int mode = upload_mode(c, pixels);
  size_t bytes = mode == B_INLINE ? image_bytes(c, 0, w, h, 1, format, type, 0) : 0;
  uint32_t *o = emit_data(c, OP_TexImage2D, 8, mode, pixels, bytes);
  o[0] = target; o[1] = (uint32_t)level; o[2] = (uint32_t)ifmt; o[3] = (uint32_t)w; o[4] = (uint32_t)h; o[5] = (uint32_t)border; o[6] = format; o[7] = type;
  done(c);
}
static void glTexParameterf(GLenum target, GLenum pname, GLfloat param) { CTX; CMD(TexParameterf, target, pname, fbits(param)); }
static void glTexParameterfv(GLenum target, GLenum pname, const GLfloat *params) { glTexParameterf(target, pname, params[0]); }
static void glTexParameteri(GLenum target, GLenum pname, GLint param) { CTX; CMD(TexParameteri, target, pname, (uint32_t)param); }
static void glTexParameteriv(GLenum target, GLenum pname, const GLint *params) { glTexParameteri(target, pname, params[0]); }
static void glTexSubImage2D(GLenum target, GLint level, GLint x, GLint y, GLsizei w, GLsizei h, GLenum format, GLenum type, const void *pixels) {
  CTX;
  int mode = upload_mode(c, pixels);
  if (mode == B_NULL) { add_error(c, GL_INVALID_OPERATION); return; }
  size_t bytes = mode == B_INLINE ? image_bytes(c, 0, w, h, 1, format, type, 0) : 0;
  uint32_t *o = emit_data(c, OP_TexSubImage2D, 8, mode, pixels, bytes);
  o[0] = target; o[1] = (uint32_t)level; o[2] = (uint32_t)x; o[3] = (uint32_t)y; o[4] = (uint32_t)w; o[5] = (uint32_t)h; o[6] = format; o[7] = type;
  done(c);
}

/* Uniforms: a location of -1 is ignored silently, as GL requires. */
static void uniform_scalar(unsigned op, GLint loc, int n, const uint32_t *v) {
  CTX;
  if (loc == -1) return;
  uint32_t *w = emit(c, op, 2 + (uint32_t)n);
  w[0] = (uint32_t)loc;
  w[1] = (uint32_t)n;
  memcpy(w + 2, v, (size_t)n * 4);
}
static void uniform_vector(unsigned op, GLint loc, int n, GLsizei count, const void *v) {
  CTX;
  if (loc == -1) return;
  if (count < 0) { add_error(c, GL_INVALID_VALUE); return; }
  uint32_t *w = emit_data(c, op, 3, B_INLINE, v, (size_t)count * (size_t)n * 4);
  w[0] = (uint32_t)loc; w[1] = (uint32_t)n; w[2] = (uint32_t)count;
  done(c);
}
static void uniform_matrix(GLint loc, int cols, int rows, GLsizei count, GLboolean transpose, const GLfloat *v) {
  CTX;
  if (loc == -1) return;
  if (count < 0) { add_error(c, GL_INVALID_VALUE); return; }
  uint32_t *w = emit_data(c, OP_UniformMatrixfv, 5, B_INLINE, v, (size_t)count * (size_t)(cols * rows) * 4);
  w[0] = (uint32_t)loc; w[1] = (uint32_t)cols; w[2] = (uint32_t)rows; w[3] = (uint32_t)count; w[4] = transpose;
  done(c);
}
static void glUniform1f(GLint l, GLfloat x) { uint32_t v[] = {fbits(x)}; uniform_scalar(OP_Uniformf, l, 1, v); }
static void glUniform2f(GLint l, GLfloat x, GLfloat y) { uint32_t v[] = {fbits(x), fbits(y)}; uniform_scalar(OP_Uniformf, l, 2, v); }
static void glUniform3f(GLint l, GLfloat x, GLfloat y, GLfloat z) { uint32_t v[] = {fbits(x), fbits(y), fbits(z)}; uniform_scalar(OP_Uniformf, l, 3, v); }
static void glUniform4f(GLint l, GLfloat x, GLfloat y, GLfloat z, GLfloat w) { uint32_t v[] = {fbits(x), fbits(y), fbits(z), fbits(w)}; uniform_scalar(OP_Uniformf, l, 4, v); }
static void glUniform1i(GLint l, GLint x) { uint32_t v[] = {(uint32_t)x}; uniform_scalar(OP_Uniformi, l, 1, v); }
static void glUniform2i(GLint l, GLint x, GLint y) { uint32_t v[] = {(uint32_t)x, (uint32_t)y}; uniform_scalar(OP_Uniformi, l, 2, v); }
static void glUniform3i(GLint l, GLint x, GLint y, GLint z) { uint32_t v[] = {(uint32_t)x, (uint32_t)y, (uint32_t)z}; uniform_scalar(OP_Uniformi, l, 3, v); }
static void glUniform4i(GLint l, GLint x, GLint y, GLint z, GLint w) { uint32_t v[] = {(uint32_t)x, (uint32_t)y, (uint32_t)z, (uint32_t)w}; uniform_scalar(OP_Uniformi, l, 4, v); }
static void glUniform1ui(GLint l, GLuint x) { uint32_t v[] = {x}; uniform_scalar(OP_Uniformui, l, 1, v); }
static void glUniform2ui(GLint l, GLuint x, GLuint y) { uint32_t v[] = {x, y}; uniform_scalar(OP_Uniformui, l, 2, v); }
static void glUniform3ui(GLint l, GLuint x, GLuint y, GLuint z) { uint32_t v[] = {x, y, z}; uniform_scalar(OP_Uniformui, l, 3, v); }
static void glUniform4ui(GLint l, GLuint x, GLuint y, GLuint z, GLuint w) { uint32_t v[] = {x, y, z, w}; uniform_scalar(OP_Uniformui, l, 4, v); }
static void glUniform1fv(GLint l, GLsizei n, const GLfloat *v) { uniform_vector(OP_Uniformfv, l, 1, n, v); }
static void glUniform2fv(GLint l, GLsizei n, const GLfloat *v) { uniform_vector(OP_Uniformfv, l, 2, n, v); }
static void glUniform3fv(GLint l, GLsizei n, const GLfloat *v) { uniform_vector(OP_Uniformfv, l, 3, n, v); }
static void glUniform4fv(GLint l, GLsizei n, const GLfloat *v) { uniform_vector(OP_Uniformfv, l, 4, n, v); }
static void glUniform1iv(GLint l, GLsizei n, const GLint *v) { uniform_vector(OP_Uniformiv, l, 1, n, v); }
static void glUniform2iv(GLint l, GLsizei n, const GLint *v) { uniform_vector(OP_Uniformiv, l, 2, n, v); }
static void glUniform3iv(GLint l, GLsizei n, const GLint *v) { uniform_vector(OP_Uniformiv, l, 3, n, v); }
static void glUniform4iv(GLint l, GLsizei n, const GLint *v) { uniform_vector(OP_Uniformiv, l, 4, n, v); }
static void glUniform1uiv(GLint l, GLsizei n, const GLuint *v) { uniform_vector(OP_Uniformuiv, l, 1, n, v); }
static void glUniform2uiv(GLint l, GLsizei n, const GLuint *v) { uniform_vector(OP_Uniformuiv, l, 2, n, v); }
static void glUniform3uiv(GLint l, GLsizei n, const GLuint *v) { uniform_vector(OP_Uniformuiv, l, 3, n, v); }
static void glUniform4uiv(GLint l, GLsizei n, const GLuint *v) { uniform_vector(OP_Uniformuiv, l, 4, n, v); }
static void glUniformMatrix2fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { uniform_matrix(l, 2, 2, n, t, v); }
static void glUniformMatrix3fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { uniform_matrix(l, 3, 3, n, t, v); }
static void glUniformMatrix4fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { uniform_matrix(l, 4, 4, n, t, v); }
static void glUniformMatrix2x3fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { uniform_matrix(l, 2, 3, n, t, v); }
static void glUniformMatrix3x2fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { uniform_matrix(l, 3, 2, n, t, v); }
static void glUniformMatrix2x4fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { uniform_matrix(l, 2, 4, n, t, v); }
static void glUniformMatrix4x2fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { uniform_matrix(l, 4, 2, n, t, v); }
static void glUniformMatrix3x4fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { uniform_matrix(l, 3, 4, n, t, v); }
static void glUniformMatrix4x3fv(GLint l, GLsizei n, GLboolean t, const GLfloat *v) { uniform_matrix(l, 4, 3, n, t, v); }
static void glUseProgram(GLuint program) {
  CTX;
  c->program = program;
  CMD(UseProgram, program);
}
static void glValidateProgram(GLuint program) {
  CTX;
  invalidate(c, program);
  CMD(ValidateProgram, program);
}
static void vertex_attrib(GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w) {
  CTX;
  CMD(VertexAttribf, index, fbits(x), fbits(y), fbits(z), fbits(w));
}
static void glVertexAttrib1f(GLuint i, GLfloat x) { vertex_attrib(i, x, 0, 0, 1); }
static void glVertexAttrib1fv(GLuint i, const GLfloat *v) { vertex_attrib(i, v[0], 0, 0, 1); }
static void glVertexAttrib2f(GLuint i, GLfloat x, GLfloat y) { vertex_attrib(i, x, y, 0, 1); }
static void glVertexAttrib2fv(GLuint i, const GLfloat *v) { vertex_attrib(i, v[0], v[1], 0, 1); }
static void glVertexAttrib3f(GLuint i, GLfloat x, GLfloat y, GLfloat z) { vertex_attrib(i, x, y, z, 1); }
static void glVertexAttrib3fv(GLuint i, const GLfloat *v) { vertex_attrib(i, v[0], v[1], v[2], 1); }
static void glVertexAttrib4f(GLuint i, GLfloat x, GLfloat y, GLfloat z, GLfloat w) { vertex_attrib(i, x, y, z, w); }
static void glVertexAttrib4fv(GLuint i, const GLfloat *v) { vertex_attrib(i, v[0], v[1], v[2], v[3]); }
static int client_array(struct glr_ctx *c, const void *pointer) {
  if (c->array_buffer || !pointer) return 0;
  static int warned;
  if (!warned++) say("client-side vertex arrays are not supported");
  add_error(c, GL_INVALID_OPERATION);
  return 1;
}
static void glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer) {
  CTX;
  if (client_array(c, pointer)) return;
  CMD(VertexAttribPointer, index, (uint32_t)size, type, normalized, (uint32_t)stride, (uint32_t)(uintptr_t)pointer);
}
static void glVertexAttribIPointer(GLuint index, GLint size, GLenum type, GLsizei stride, const void *pointer) {
  CTX;
  if (client_array(c, pointer)) return;
  CMD(VertexAttribIPointer, index, (uint32_t)size, type, (uint32_t)stride, (uint32_t)(uintptr_t)pointer);
}
static void glViewport(GLint x, GLint y, GLsizei w, GLsizei h) {
  CTX;
  c->viewport[0] = x; c->viewport[1] = y; c->viewport[2] = w; c->viewport[3] = h;
  CMD(Viewport, (uint32_t)x, (uint32_t)y, (uint32_t)w, (uint32_t)h);
}

/* ES 3.0 additions */
static void glReadBuffer(GLenum src) { CTX; CMD(ReadBuffer, src); }
static void glDrawRangeElements(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices) {
  CTX;
  int scratch;
  uint32_t offset = index_offset(c, count, type, indices, &scratch);
  CMD(DrawRangeElements, mode, start, end, (uint32_t)count, type, offset);
  index_restore(c, scratch);
}
static void glTexImage3D(GLenum target, GLint level, GLint ifmt, GLsizei w, GLsizei h, GLsizei d, GLint border, GLenum format, GLenum type, const void *pixels) {
  CTX;
  int mode = upload_mode(c, pixels);
  size_t bytes = mode == B_INLINE ? image_bytes(c, 0, w, h, d, format, type, 1) : 0;
  uint32_t *o = emit_data(c, OP_TexImage3D, 9, mode, pixels, bytes);
  o[0] = target; o[1] = (uint32_t)level; o[2] = (uint32_t)ifmt; o[3] = (uint32_t)w; o[4] = (uint32_t)h; o[5] = (uint32_t)d;
  o[6] = (uint32_t)border; o[7] = format; o[8] = type;
  done(c);
}
static void glTexSubImage3D(GLenum target, GLint level, GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d, GLenum format, GLenum type, const void *pixels) {
  CTX;
  int mode = upload_mode(c, pixels);
  if (mode == B_NULL) { add_error(c, GL_INVALID_OPERATION); return; }
  size_t bytes = mode == B_INLINE ? image_bytes(c, 0, w, h, d, format, type, 1) : 0;
  uint32_t *o = emit_data(c, OP_TexSubImage3D, 10, mode, pixels, bytes);
  o[0] = target; o[1] = (uint32_t)level; o[2] = (uint32_t)x; o[3] = (uint32_t)y; o[4] = (uint32_t)z;
  o[5] = (uint32_t)w; o[6] = (uint32_t)h; o[7] = (uint32_t)d; o[8] = format; o[9] = type;
  done(c);
}
static void glCopyTexSubImage3D(GLenum target, GLint level, GLint xo, GLint yo, GLint zo, GLint x, GLint y, GLsizei w, GLsizei h) {
  CTX;
  CMD(CopyTexSubImage3D, target, (uint32_t)level, (uint32_t)xo, (uint32_t)yo, (uint32_t)zo, (uint32_t)x, (uint32_t)y, (uint32_t)w, (uint32_t)h);
}
static void glCompressedTexImage3D(GLenum target, GLint level, GLenum ifmt, GLsizei w, GLsizei h, GLsizei d, GLint border, GLsizei size, const void *data) {
  CTX;
  uint32_t *o = emit_data(c, OP_CompressedTexImage3D, 7, upload_mode(c, data), data, (size_t)(size > 0 ? size : 0));
  o[0] = target; o[1] = (uint32_t)level; o[2] = ifmt; o[3] = (uint32_t)w; o[4] = (uint32_t)h; o[5] = (uint32_t)d; o[6] = (uint32_t)border;
  done(c);
}
static void glCompressedTexSubImage3D(GLenum target, GLint level, GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d, GLenum format, GLsizei size, const void *data) {
  CTX;
  uint32_t *o = emit_data(c, OP_CompressedTexSubImage3D, 9, upload_mode(c, data), data, (size_t)(size > 0 ? size : 0));
  o[0] = target; o[1] = (uint32_t)level; o[2] = (uint32_t)x; o[3] = (uint32_t)y; o[4] = (uint32_t)z;
  o[5] = (uint32_t)w; o[6] = (uint32_t)h; o[7] = (uint32_t)d; o[8] = format;
  done(c);
}
static void glBeginQuery(GLenum target, GLuint id) { CTX; CMD(BeginQuery, target, id); }
static void glEndQuery(GLenum target) { CTX; CMD(EndQuery, target); }
static void glGetQueryiv(GLenum target, GLenum pname, GLint *params) {
  CTX;
  double v[1];
  if (q2(c, OP_GetQuery, target, pname, 2, v, 1) >= 1) params[0] = to_int(v[0], 0);
}
static void glGetQueryObjectuiv(GLuint id, GLenum pname, GLuint *params) {
  CTX;
  double v[1] = {0};
  if (pname == GL_QUERY_RESULT) {
    /* WebGL results become available only after the host returns to its event
     * loop; GL's QUERY_RESULT waits, so poll availability. */
    for (int i = 0; i < 4000; i++) {
      if (q2(c, OP_GetQueryParameter, id, GL_QUERY_RESULT_AVAILABLE, 2, v, 1) < 1 || v[0]) break;
      struct timespec ts = {0, 500000};
      nanosleep(&ts, NULL);
    }
  }
  if (q2(c, OP_GetQueryParameter, id, pname, 2, v, 1) >= 1) params[0] = (GLuint)v[0];
}
static GLboolean glUnmapBuffer(GLenum target) {
  CTXR(GL_FALSE);
  struct mapping *m = mapping_of(c, target);
  if (!m) { add_error(c, GL_INVALID_OPERATION); return GL_FALSE; }
  if ((m->access & GL_MAP_WRITE_BIT) && !(m->access & GL_MAP_FLUSH_EXPLICIT_BIT) && m->length) {
    shadow_sub(c, target, (size_t)m->offset, (size_t)m->length, m->ptr);
    size_t bytes = (size_t)m->length;
    if (bytes > INLINE_MAX) {
      /* The mapped block itself goes with the batch: no copy. */
      if (c->staged_bytes + bytes > STAGE_MAX) flush(c);
      uint32_t *w = emit(c, OP_BufferSubData, 2 + 3); /* may run the batch: stage after it */
      w[0] = target; w[1] = (uint32_t)m->offset; w[2] = (uint32_t)bytes; w[3] = B_POINTER;
      w[4] = (uint32_t)(uintptr_t)m->ptr;
      if (stage(c, m->ptr, bytes)) m->ptr = NULL;
      else { c->must_flush = 1; c->prof.flush_ops[OP_BufferSubData]++; c->prof.end_op = OP_BufferSubData; }
    } else {
      uint32_t *w = emit_data(c, OP_BufferSubData, 2, B_INLINE, m->ptr, bytes);
      w[0] = target; w[1] = (uint32_t)m->offset;
    }
  }
  done(c);
  block_release(c, m->ptr);
  m->ptr = NULL;
  return GL_TRUE;
}
static void glDrawBuffers(GLsizei n, const GLenum *bufs) {
  CTX;
  if (n < 0) { add_error(c, GL_INVALID_VALUE); return; }
  uint32_t *w = emit(c, OP_DrawBuffers, 1 + (uint32_t)n);
  w[0] = (uint32_t)n;
  memcpy(w + 1, bufs, (size_t)n * 4);
}
static void glBlitFramebuffer(GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1, GLbitfield mask, GLenum filter) {
  CTX;
  CMD(BlitFramebuffer, (uint32_t)sx0, (uint32_t)sy0, (uint32_t)sx1, (uint32_t)sy1, (uint32_t)dx0, (uint32_t)dy0, (uint32_t)dx1, (uint32_t)dy1, mask, filter);
}
static void glRenderbufferStorageMultisample(GLenum target, GLsizei samples, GLenum ifmt, GLsizei w, GLsizei h) {
  CTX;
  CMD(RenderbufferStorageMultisample, target, (uint32_t)samples, ifmt, (uint32_t)w, (uint32_t)h);
}
static void glFramebufferTextureLayer(GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer) {
  CTX;
  CMD(FramebufferTextureLayer, target, attachment, texture, (uint32_t)level, (uint32_t)layer);
}
/* Mapped ranges live in guest memory; written ranges are uploaded on
 * FlushMappedBufferRange/UnmapBuffer, read access fetches the contents. */
static void *glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access) {
  CTXR(NULL);
  if (offset < 0 || length <= 0 || !(access & (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT))) { add_error(c, GL_INVALID_VALUE); return NULL; }
  GLuint *slot = buffer_slot(c, target);
  if (!slot || !*slot || mapping_of(c, target)) { add_error(c, GL_INVALID_OPERATION); return NULL; }
  struct mapping *m = NULL;
  for (int i = 0; i < 8 && !m; i++)
    if (!c->maps[i].ptr) m = &c->maps[i];
  if (!m) { add_error(c, GL_OUT_OF_MEMORY); return NULL; }
  void *ptr = block_alloc(c, (size_t)length);
  if (!ptr) { add_error(c, GL_OUT_OF_MEMORY); return NULL; }
  int fetch = (access & GL_MAP_READ_BIT) ||
              !(access & (GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_FLUSH_EXPLICIT_BIT));
  struct gbuf *g = bound_gbuf(c, target);
  if (fetch && g && g->shadow && (size_t)offset <= g->size && (size_t)length <= g->size - (size_t)offset) {
    memcpy(ptr, g->shadow + offset, (size_t)length);
  } else if (fetch) {
    uint32_t start;
    uint32_t *w = emit_query(c, OP_GetBufferSubData, 4 + 1, &start);
    w[0] = target; w[1] = (uint32_t)offset; w[2] = (uint32_t)length; w[3] = B_POINTER; w[4] = (uint32_t)(uintptr_t)ptr;
    if (query(c, start) < 0) memset(ptr, 0, (size_t)length);
  }
  *m = (struct mapping){.target = target, .buffer = *slot, .ptr = ptr, .offset = offset, .length = length, .access = access};
  return ptr;
}
static void glFlushMappedBufferRange(GLenum target, GLintptr offset, GLsizeiptr length) {
  CTX;
  struct mapping *m = mapping_of(c, target);
  if (!m || !(m->access & GL_MAP_FLUSH_EXPLICIT_BIT)) { add_error(c, GL_INVALID_OPERATION); return; }
  if (offset < 0 || length < 0 || offset + length > m->length) { add_error(c, GL_INVALID_VALUE); return; }
  if (!length) return;
  uint32_t *w = emit_data(c, OP_BufferSubData, 2, B_INLINE, (char *)m->ptr + offset, (size_t)length);
  w[0] = target; w[1] = (uint32_t)(m->offset + offset);
  shadow_sub(c, target, (size_t)(m->offset + offset), (size_t)length, (char *)m->ptr + offset);
  done(c);
}
static void glBindVertexArray(GLuint array) {
  CTX;
  c->vao = array;
  CMD(BindVertexArray, array);
}
static void glBeginTransformFeedback(GLenum mode) { CTX; CMD(BeginTransformFeedback, mode); }
static void glEndTransformFeedback(void) { CTX; CMD0(EndTransformFeedback); }
/* Indexed binds also set the generic binding. */
static void indexed_bind(struct glr_ctx *c, GLenum target, GLuint buffer) {
  if (target == GL_UNIFORM_BUFFER) c->uniform_buffer = buffer;
  if (target == GL_TRANSFORM_FEEDBACK_BUFFER) {
    c->tf_buffer = buffer;
    host_writes(c, buffer);
  }
}
static void glBindBufferRange(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size) {
  CTX;
  indexed_bind(c, target, buffer);
  CMD(BindBufferRange, target, index, buffer, (uint32_t)offset, (uint32_t)size);
}
static void glBindBufferBase(GLenum target, GLuint index, GLuint buffer) {
  CTX;
  indexed_bind(c, target, buffer);
  CMD(BindBufferBase, target, index, buffer);
}
static void glTransformFeedbackVaryings(GLuint program, GLsizei count, const GLchar *const *varyings, GLenum mode) {
  CTX;
  if (count < 0) { add_error(c, GL_INVALID_VALUE); return; }
  size_t total = 0;
  for (GLsizei i = 0; i < count; i++) total += strlen(varyings[i]) + 1;
  char *names = malloc(total ? total : 1);
  if (!names) { add_error(c, GL_OUT_OF_MEMORY); return; }
  size_t used = 0;
  for (GLsizei i = 0; i < count; i++) {
    size_t l = strlen(varyings[i]) + 1;
    memcpy(names + used, varyings[i], l);
    used += l;
  }
  invalidate(c, program);
  uint32_t *w = emit_data(c, OP_TransformFeedbackVaryings, 2, B_INLINE, names, used);
  w[0] = program; w[1] = mode;
  done(c);
  free(names);
}
static void glGetTransformFeedbackVarying(GLuint program, GLuint index, GLsizei size, GLsizei *length, GLsizei *vsize, GLenum *type, GLchar *name) {
  CTX;
  struct obj *o = program_info(c, program);
  if (!o) return;
  if (index >= (GLuint)o->nv) { add_error(c, GL_INVALID_VALUE); return; }
  copy_name(o->v[index].name, size, length, name);
  if (vsize) *vsize = o->v[index].size;
  if (type) *type = o->v[index].type;
}
static void glVertexAttribI4i(GLuint i, GLint x, GLint y, GLint z, GLint w) { CTX; CMD(VertexAttribI4i, i, (uint32_t)x, (uint32_t)y, (uint32_t)z, (uint32_t)w); }
static void glVertexAttribI4ui(GLuint i, GLuint x, GLuint y, GLuint z, GLuint w) { CTX; CMD(VertexAttribI4ui, i, x, y, z, w); }
static void glVertexAttribI4iv(GLuint i, const GLint *v) { glVertexAttribI4i(i, v[0], v[1], v[2], v[3]); }
static void glVertexAttribI4uiv(GLuint i, const GLuint *v) { glVertexAttribI4ui(i, v[0], v[1], v[2], v[3]); }
static GLint glGetFragDataLocation(GLuint program, const GLchar *name) {
  CTXR(-1);
  if (!name) return -1;
  uint32_t start;
  if (c->used + 1 + 3 + (strlen(name) + 3) / 4 > CMD_WORDS) flush(c);
  start = c->used;
  uint32_t *w = emit_data(c, OP_GetFragDataLocation, 1, B_INLINE, name, strlen(name));
  w[0] = program;
  double v[1];
  return query_numbers(c, start, v, 1) >= 1 ? (GLint)v[0] : -1;
}
static void clear_buffer(unsigned op, GLenum buffer, GLint drawbuffer, const void *value) {
  CTX;
  int n = buffer == GL_COLOR ? 4 : 1;
  uint32_t *w = emit(c, op, 2 + 4);
  w[0] = buffer; w[1] = (uint32_t)drawbuffer;
  memset(w + 2, 0, 16);
  memcpy(w + 2, value, (size_t)n * 4);
}
static void glClearBufferiv(GLenum b, GLint d, const GLint *v) { clear_buffer(OP_ClearBufferiv, b, d, v); }
static void glClearBufferuiv(GLenum b, GLint d, const GLuint *v) { clear_buffer(OP_ClearBufferuiv, b, d, v); }
static void glClearBufferfv(GLenum b, GLint d, const GLfloat *v) { clear_buffer(OP_ClearBufferfv, b, d, v); }
static void glClearBufferfi(GLenum b, GLint d, GLfloat depth, GLint stencil) { CTX; CMD(ClearBufferfi, b, (uint32_t)d, fbits(depth), (uint32_t)stencil); }
static void glCopyBufferSubData(GLenum rt, GLenum wt, GLintptr ro, GLintptr wo, GLsizeiptr size) {
  CTX;
  GLuint *slot = buffer_slot(c, wt);
  if (slot) drop_shadow(c, gbuf_of(c, *slot));
  CMD(CopyBufferSubData, rt, wt, (uint32_t)ro, (uint32_t)wo, (uint32_t)size);
}
static int uniform_index(const struct obj *o, const char *name) {
  size_t len = strlen(name);
  for (int i = 0; i < o->nu; i++) {
    const char *u = o->u[i].name;
    size_t base = strlen(u);
    if (!strcmp(u, name)) return i;
    if (base >= 3 && !strcmp(u + base - 3, "[0]") && len == base - 3 && !strncmp(u, name, len)) return i;
  }
  return -1;
}
static void glGetUniformIndices(GLuint program, GLsizei count, const GLchar *const *names, GLuint *indices) {
  CTX;
  struct obj *o = program_info(c, program);
  for (GLsizei i = 0; i < count; i++) {
    int k = o ? uniform_index(o, names[i]) : -1;
    indices[i] = k < 0 ? GL_INVALID_INDEX : (GLuint)k;
  }
}
static void glGetActiveUniformsiv(GLuint program, GLsizei count, const GLuint *indices, GLenum pname, GLint *params) {
  CTX;
  struct obj *o = program_info(c, program);
  if (!o) return;
  for (GLsizei i = 0; i < count; i++)
    if (indices[i] >= (GLuint)o->nu) { add_error(c, GL_INVALID_VALUE); return; }
  for (GLsizei i = 0; i < count; i++) {
    const struct var *u = &o->u[indices[i]];
    switch (pname) {
    case GL_UNIFORM_TYPE: params[i] = (GLint)u->type; break;
    case GL_UNIFORM_SIZE: params[i] = u->size; break;
    case GL_UNIFORM_NAME_LENGTH: params[i] = (GLint)strlen(u->name) + 1; break;
    case GL_UNIFORM_BLOCK_INDEX: params[i] = u->block; break;
    case GL_UNIFORM_OFFSET: params[i] = u->offset; break;
    case GL_UNIFORM_ARRAY_STRIDE: params[i] = u->astride; break;
    case GL_UNIFORM_MATRIX_STRIDE: params[i] = u->mstride; break;
    case GL_UNIFORM_IS_ROW_MAJOR: params[i] = u->rowmajor; break;
    default: add_error(c, GL_INVALID_ENUM); return;
    }
  }
}
static GLuint glGetUniformBlockIndex(GLuint program, const GLchar *name) {
  CTXR(GL_INVALID_INDEX);
  struct obj *o = program_info(c, program);
  if (!o || !name) return GL_INVALID_INDEX;
  for (int i = 0; i < o->nb; i++)
    if (!strcmp(o->b[i].name, name)) return (GLuint)i;
  return GL_INVALID_INDEX;
}
static void glGetActiveUniformBlockiv(GLuint program, GLuint index, GLenum pname, GLint *params) {
  CTX;
  struct obj *o = program_info(c, program);
  if (!o) return;
  if (index >= (GLuint)o->nb) { add_error(c, GL_INVALID_VALUE); return; }
  const struct block *b = &o->b[index];
  switch (pname) {
  case GL_UNIFORM_BLOCK_BINDING: params[0] = b->binding; return;
  case GL_UNIFORM_BLOCK_DATA_SIZE: params[0] = b->datasize; return;
  case GL_UNIFORM_BLOCK_NAME_LENGTH: params[0] = (GLint)strlen(b->name) + 1; return;
  case GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS: params[0] = b->n; return;
  case GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES: memcpy(params, b->indices, (size_t)b->n * sizeof *params); return;
  case GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER: params[0] = b->refvs; return;
  case GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER: params[0] = b->reffs; return;
  }
  add_error(c, GL_INVALID_ENUM);
}
static void glGetActiveUniformBlockName(GLuint program, GLuint index, GLsizei size, GLsizei *length, GLchar *name) {
  CTX;
  struct obj *o = program_info(c, program);
  if (!o) return;
  if (index >= (GLuint)o->nb) { add_error(c, GL_INVALID_VALUE); return; }
  copy_name(o->b[index].name, size, length, name);
}
static void glUniformBlockBinding(GLuint program, GLuint index, GLuint binding) {
  CTX;
  struct obj *o = object(c, program);
  if (o && o->info_valid && index < (GLuint)o->nb) o->b[index].binding = (GLint)binding;
  CMD(UniformBlockBinding, program, index, binding);
}
static void glDrawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instances) {
  CTX;
  CMD(DrawArraysInstanced, mode, (uint32_t)first, (uint32_t)count, (uint32_t)instances);
}
static void glDrawElementsInstanced(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instances) {
  CTX;
  int scratch;
  uint32_t offset = index_offset(c, count, type, indices, &scratch);
  CMD(DrawElementsInstanced, mode, (uint32_t)count, type, offset, (uint32_t)instances);
  index_restore(c, scratch);
}
/* Commands run in order on the host and every result is read back through a
 * later synchronous call, so a fence is complete once its batch was sent. */
static GLsync glFenceSync(GLenum condition, GLbitfield flags) {
  CTXR(NULL);
  if (condition != GL_SYNC_GPU_COMMANDS_COMPLETE || flags) { add_error(c, condition != GL_SYNC_GPU_COMMANDS_COMPLETE ? GL_INVALID_ENUM : GL_INVALID_VALUE); return NULL; }
  struct glr_sync *s = calloc(1, sizeof *s);
  if (!s) { add_error(c, GL_OUT_OF_MEMORY); return NULL; }
  s->magic = SYNC_MAGIC;
  s->ctx = c;
  return (GLsync)s;
}
static int valid_sync(GLsync sync) { return sync && ((struct glr_sync *)sync)->magic == SYNC_MAGIC; }
static GLboolean glIsSync(GLsync sync) { return valid_sync(sync); }
static void glDeleteSync(GLsync sync) {
  if (!sync) return;
  if (!valid_sync(sync)) { CTX; add_error(c, GL_INVALID_VALUE); return; }
  ((struct glr_sync *)sync)->magic = 0;
  free(sync);
}
static GLenum glClientWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout) {
  (void)flags; (void)timeout;
  CTXR(GL_WAIT_FAILED);
  if (!valid_sync(sync)) { add_error(c, GL_INVALID_VALUE); return GL_WAIT_FAILED; }
  if (!c->used) return GL_ALREADY_SIGNALED;
  flush(c);
  return GL_CONDITION_SATISFIED;
}
static void glWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout) {
  (void)flags; (void)timeout;
  CTX;
  if (!valid_sync(sync)) add_error(c, GL_INVALID_VALUE);
}
static void glGetSynciv(GLsync sync, GLenum pname, GLsizei count, GLsizei *length, GLint *values) {
  CTX;
  if (!valid_sync(sync)) { add_error(c, GL_INVALID_VALUE); return; }
  GLint v;
  switch (pname) {
  case GL_OBJECT_TYPE: v = GL_SYNC_FENCE; break;
  case GL_SYNC_STATUS: flush(c); v = GL_SIGNALED; break;
  case GL_SYNC_CONDITION: v = GL_SYNC_GPU_COMMANDS_COMPLETE; break;
  case GL_SYNC_FLAGS: v = 0; break;
  default: add_error(c, GL_INVALID_ENUM); return;
  }
  if (count > 0) values[0] = v;
  if (length) *length = count > 0;
}
static void glBindSampler(GLuint unit, GLuint sampler) {
  CTX;
  if (unit < MAX_UNITS) c->sampler[unit] = sampler;
  CMD(BindSampler, unit, sampler);
}
static void glSamplerParameteri(GLuint s, GLenum pname, GLint param) { CTX; CMD(SamplerParameteri, s, pname, (uint32_t)param); }
static void glSamplerParameteriv(GLuint s, GLenum pname, const GLint *param) { glSamplerParameteri(s, pname, param[0]); }
static void glSamplerParameterf(GLuint s, GLenum pname, GLfloat param) { CTX; CMD(SamplerParameterf, s, pname, fbits(param)); }
static void glSamplerParameterfv(GLuint s, GLenum pname, const GLfloat *param) { glSamplerParameterf(s, pname, param[0]); }
static void glGetSamplerParameteriv(GLuint s, GLenum pname, GLint *params) {
  CTX;
  double v[1];
  if (q2(c, OP_GetSamplerParameter, s, pname, 2, v, 1) >= 1) params[0] = to_int(v[0], 0);
}
static void glGetSamplerParameterfv(GLuint s, GLenum pname, GLfloat *params) {
  CTX;
  double v[1];
  if (q2(c, OP_GetSamplerParameter, s, pname, 2, v, 1) >= 1) params[0] = (GLfloat)v[0];
}
static void glVertexAttribDivisor(GLuint index, GLuint divisor) { CTX; CMD(VertexAttribDivisor, index, divisor); }
static void glBindTransformFeedback(GLenum target, GLuint id) {
  CTX;
  if (target == GL_TRANSFORM_FEEDBACK) c->tfo = id;
  CMD(BindTransformFeedback, target, id);
}
static void glPauseTransformFeedback(void) { CTX; CMD0(PauseTransformFeedback); }
static void glResumeTransformFeedback(void) { CTX; CMD0(ResumeTransformFeedback); }
static void glGetProgramBinary(GLuint program, GLsizei size, GLsizei *length, GLenum *format, void *binary) {
  (void)program; (void)size; (void)format; (void)binary;
  CTX;
  if (length) *length = 0;
  add_error(c, GL_INVALID_OPERATION);
}
static void glProgramBinary(GLuint program, GLenum format, const void *binary, GLsizei length) {
  (void)program; (void)format; (void)binary; (void)length;
  CTX;
  add_error(c, GL_INVALID_ENUM);
}
static void glProgramParameteri(GLuint program, GLenum pname, GLint value) { (void)program; (void)pname; (void)value; }
static void glInvalidateFramebuffer(GLenum target, GLsizei n, const GLenum *attachments) {
  CTX;
  if (n < 0) { add_error(c, GL_INVALID_VALUE); return; }
  uint32_t *w = emit(c, OP_InvalidateFramebuffer, 2 + (uint32_t)n);
  w[0] = target; w[1] = (uint32_t)n;
  memcpy(w + 2, attachments, (size_t)n * 4);
}
static void glInvalidateSubFramebuffer(GLenum target, GLsizei n, const GLenum *attachments, GLint x, GLint y, GLsizei width, GLsizei height) {
  CTX;
  if (n < 0) { add_error(c, GL_INVALID_VALUE); return; }
  uint32_t *w = emit(c, OP_InvalidateSubFramebuffer, 6 + (uint32_t)n);
  w[0] = target; w[1] = (uint32_t)x; w[2] = (uint32_t)y; w[3] = (uint32_t)width; w[4] = (uint32_t)height; w[5] = (uint32_t)n;
  memcpy(w + 6, attachments, (size_t)n * 4);
}
static void glTexStorage2D(GLenum target, GLsizei levels, GLenum ifmt, GLsizei w, GLsizei h) {
  CTX;
  CMD(TexStorage2D, target, (uint32_t)levels, ifmt, (uint32_t)w, (uint32_t)h);
}
static void glTexStorage3D(GLenum target, GLsizei levels, GLenum ifmt, GLsizei w, GLsizei h, GLsizei d) {
  CTX;
  CMD(TexStorage3D, target, (uint32_t)levels, ifmt, (uint32_t)w, (uint32_t)h, (uint32_t)d);
}
static void glGetInternalformativ(GLenum target, GLenum ifmt, GLenum pname, GLsizei count, GLint *params) {
  CTX;
  double v[64];
  int n = q3(c, OP_GetInternalformatParameter, target, ifmt, pname, v, 64);
  for (int i = 0; i < n && i < count && i < 64; i++) params[i] = (GLint)v[i];
}

/* ==== EGL 1.5 ==== */

struct glr_config { EGLint id, red, green, blue, alpha, depth, stencil; };
static const struct glr_config configs[] = {
  {1, 8, 8, 8, 8, 24, 8}, {2, 8, 8, 8, 8, 0, 0}, {3, 8, 8, 8, 0, 24, 8}, {4, 8, 8, 8, 0, 0, 0},
};
#define NCONFIGS ((EGLint)(sizeof configs / sizeof configs[0]))

struct glr_surface {
  uint32_t magic;
  EGLint width, height, config_id;
  int destroy_pending, bound;
  struct glr_surface *next;
  /* window surfaces: native window and the swap readback buffer */
  int window;
  unsigned long native;
  uint32_t *pixels;
  size_t pixels_cap;
  int presented; /* the window shows a whole frame of this size */
  /* What the window shows (top row first), when shown_w/shown_h match the
   * surface; a swap sends the window only the box that differs from it. */
  uint32_t *shown_px;
  EGLint shown_w, shown_h;
  /* Keyed presentation: the key colour the window is filled with (0: none),
   * and swaps left before trying again after the host declined or the window
   * could not be keyed. */
  unsigned key;
  int key_wait;
};
#define SURFACE_MAGIC 0x676c7366u

static struct { int initialized; } display;
#define DISPLAY ((EGLDisplay)&display)
static pthread_mutex_t egl_lock = PTHREAD_MUTEX_INITIALIZER;
static struct glr_ctx *contexts;
static struct glr_surface *surfaces;
static _Thread_local EGLint tls_error = EGL_SUCCESS;
static _Thread_local EGLenum tls_api = EGL_OPENGL_ES_API;
static const struct glremote_window_ops *window_ops;

void glremote_set_window_ops(const struct glremote_window_ops *ops) { window_ops = ops; }

static EGLBoolean fail(EGLint error) {
  tls_error = error;
  return EGL_FALSE;
}
static EGLBoolean ok(void) {
  tls_error = EGL_SUCCESS;
  return EGL_TRUE;
}
static int valid_display(EGLDisplay dpy, int need_init) {
  if (dpy != DISPLAY) { tls_error = EGL_BAD_DISPLAY; return 0; }
  if (need_init && !display.initialized) { tls_error = EGL_NOT_INITIALIZED; return 0; }
  return 1;
}
static const struct glr_config *config_of(EGLConfig config) {
  intptr_t id = (intptr_t)config;
  return id >= 1 && id <= NCONFIGS ? &configs[id - 1] : NULL;
}
static struct glr_ctx *context_of(EGLContext ctx) {
  for (struct glr_ctx *c = contexts; c; c = c->next)
    if ((EGLContext)c == ctx) return c;
  return NULL;
}
static struct glr_surface *surface_of(EGLSurface surface) {
  for (struct glr_surface *s = surfaces; s; s = s->next)
    if ((EGLSurface)s == surface) return s;
  return NULL;
}

static EGLDisplay eglGetDisplay(EGLNativeDisplayType native) {
  (void)native;
  return DISPLAY;
}
static EGLDisplay eglGetPlatformDisplay(EGLenum platform, void *native, const EGLAttrib *attribs) {
  (void)platform; (void)native; (void)attribs;
  return DISPLAY;
}
static EGLDisplay eglGetPlatformDisplayEXT(EGLenum platform, void *native, const EGLint *attribs) {
  (void)platform; (void)native; (void)attribs;
  return DISPLAY;
}
static EGLBoolean eglInitialize(EGLDisplay dpy, EGLint *major, EGLint *minor) {
  if (!valid_display(dpy, 0)) return EGL_FALSE;
  if (!glremote_available()) {
    say("host GL remote unavailable (no WebGL2 owner)");
    return fail(EGL_NOT_INITIALIZED);
  }
  display.initialized = 1;
  if (major) *major = 1;
  if (minor) *minor = 5;
  return ok();
}
static EGLBoolean eglTerminate(EGLDisplay dpy) {
  if (!valid_display(dpy, 0)) return EGL_FALSE;
  display.initialized = 0;
  return ok();
}
static const char *eglQueryString(EGLDisplay dpy, EGLint name) {
  if (dpy == EGL_NO_DISPLAY && name == EGL_EXTENSIONS) {
    tls_error = EGL_SUCCESS;
    return "EGL_EXT_client_extensions EGL_EXT_platform_base EGL_MESA_platform_surfaceless EGL_EXT_device_base "
           "EGL_EXT_device_query";
  }
  if (!valid_display(dpy, 1)) return NULL;
  tls_error = EGL_SUCCESS;
  switch (name) {
  case EGL_VENDOR: return "glremote";
  case EGL_VERSION: return "1.5 glremote";
  case EGL_CLIENT_APIS: return "OpenGL_ES";
  case EGL_EXTENSIONS:
    return "EGL_KHR_create_context EGL_KHR_surfaceless_context EGL_KHR_no_config_context EGL_EXT_create_context_robustness"
           " EGL_EXT_buffer_age EGL_KHR_swap_buffers_with_damage EGL_EXT_swap_buffers_with_damage";
  }
  tls_error = EGL_BAD_PARAMETER;
  return NULL;
}
/* EGL_EXT_device_query: one device, the host browser's WebGL2 GPU. It is
 * reported as software (EGL_MESA_device_software, the extension Mesa and
 * Firefox use) when a host context named a software renderer. */
static struct { int unused; } device;
#define DEVICE ((EGLDeviceEXT)&device)
static int host_software;
static int software_renderer(const char *renderer) {
  static const char *const names[] = {"SwiftShader", "llvmpipe", "softpipe", "Software", "Basic Render"};
  for (size_t i = 0; i < sizeof names / sizeof names[0]; i++)
    if (strstr(renderer, names[i])) return 1;
  return 0;
}
static EGLBoolean eglQueryDisplayAttribEXT(EGLDisplay dpy, EGLint attribute, EGLAttrib *value) {
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  if (attribute != EGL_DEVICE_EXT) return fail(EGL_BAD_ATTRIBUTE);
  *value = (EGLAttrib)DEVICE;
  return ok();
}
static const char *eglQueryDeviceStringEXT(EGLDeviceEXT dev, EGLint name) {
  if (dev != DEVICE) { tls_error = EGL_BAD_DEVICE_EXT; return NULL; }
  if (name != EGL_EXTENSIONS) { tls_error = EGL_BAD_PARAMETER; return NULL; }
  tls_error = EGL_SUCCESS;
  return host_software ? "EGL_MESA_device_software" : "";
}
static EGLBoolean eglQueryDeviceAttribEXT(EGLDeviceEXT dev, EGLint attribute, EGLAttrib *value) {
  (void)attribute; (void)value;
  return fail(dev == DEVICE ? EGL_BAD_ATTRIBUTE : EGL_BAD_DEVICE_EXT);
}
static EGLint config_attrib(const struct glr_config *k, EGLint attribute, int *known) {
  *known = 1;
  switch (attribute) {
  case EGL_BUFFER_SIZE: return k->red + k->green + k->blue + k->alpha;
  case EGL_RED_SIZE: return k->red;
  case EGL_GREEN_SIZE: return k->green;
  case EGL_BLUE_SIZE: return k->blue;
  case EGL_ALPHA_SIZE: return k->alpha;
  case EGL_DEPTH_SIZE: return k->depth;
  case EGL_STENCIL_SIZE: return k->stencil;
  case EGL_CONFIG_ID: return k->id;
  case EGL_CONFIG_CAVEAT: return EGL_NONE;
  case EGL_NATIVE_VISUAL_ID: return window_ops ? window_ops->visual(k->alpha ? 32 : 24) : 0;
  case EGL_LEVEL: case EGL_SAMPLES: case EGL_SAMPLE_BUFFERS: case EGL_LUMINANCE_SIZE:
  case EGL_ALPHA_MASK_SIZE: case EGL_MIN_SWAP_INTERVAL: case EGL_TRANSPARENT_RED_VALUE: case EGL_TRANSPARENT_GREEN_VALUE:
  case EGL_TRANSPARENT_BLUE_VALUE: return 0;
  case EGL_MAX_SWAP_INTERVAL: return 1;
  case EGL_NATIVE_VISUAL_TYPE: return EGL_NONE;
  case EGL_NATIVE_RENDERABLE: case EGL_BIND_TO_TEXTURE_RGB: case EGL_BIND_TO_TEXTURE_RGBA: return EGL_FALSE;
  case EGL_SURFACE_TYPE: return EGL_PBUFFER_BIT | (window_ops ? EGL_WINDOW_BIT : 0);
  case EGL_RENDERABLE_TYPE: case EGL_CONFORMANT: return EGL_OPENGL_ES2_BIT | EGL_OPENGL_ES3_BIT;
  case EGL_COLOR_BUFFER_TYPE: return EGL_RGB_BUFFER;
  case EGL_TRANSPARENT_TYPE: return EGL_NONE;
  case EGL_MAX_PBUFFER_WIDTH: case EGL_MAX_PBUFFER_HEIGHT: return 16384;
  case EGL_MAX_PBUFFER_PIXELS: return 16384 * 16384;
  }
  *known = 0;
  return 0;
}
static EGLBoolean eglGetConfigAttrib(EGLDisplay dpy, EGLConfig config, EGLint attribute, EGLint *value) {
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  const struct glr_config *k = config_of(config);
  if (!k) return fail(EGL_BAD_CONFIG);
  int known;
  EGLint v = config_attrib(k, attribute, &known);
  if (!known) return fail(EGL_BAD_ATTRIBUTE);
  *value = v;
  return ok();
}
static EGLBoolean eglGetConfigs(EGLDisplay dpy, EGLConfig *out, EGLint size, EGLint *count) {
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  if (!count) return fail(EGL_BAD_PARAMETER);
  EGLint n = 0;
  if (!out) n = NCONFIGS;
  else
    for (; n < NCONFIGS && n < size; n++) out[n] = (EGLConfig)(intptr_t)configs[n].id;
  *count = n;
  return ok();
}
static const EGLint *choose_request;
static int choose_compare(const void *pa, const void *pb) {
  const struct glr_config *a = *(const struct glr_config *const *)pa, *b = *(const struct glr_config *const *)pb;
  int want_r = 0, want_g = 0, want_b = 0, want_a = 0;
  for (const EGLint *p = choose_request; p && p[0] != EGL_NONE; p += 2) {
    if (p[1] == EGL_DONT_CARE || p[1] == 0) continue;
    if (p[0] == EGL_RED_SIZE) want_r = 1;
    if (p[0] == EGL_GREEN_SIZE) want_g = 1;
    if (p[0] == EGL_BLUE_SIZE) want_b = 1;
    if (p[0] == EGL_ALPHA_SIZE) want_a = 1;
  }
  int sa = want_r * a->red + want_g * a->green + want_b * a->blue + want_a * a->alpha;
  int sb = want_r * b->red + want_g * b->green + want_b * b->blue + want_a * b->alpha;
  if (sa != sb) return sb - sa;
  int ba = a->red + a->green + a->blue + a->alpha, bb = b->red + b->green + b->blue + b->alpha;
  if (ba != bb) return ba - bb;
  if (a->depth != b->depth) return a->depth - b->depth;
  if (a->stencil != b->stencil) return a->stencil - b->stencil;
  return a->id - b->id;
}
static EGLBoolean eglChooseConfig(EGLDisplay dpy, const EGLint *attribs, EGLConfig *out, EGLint size, EGLint *count) {
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  if (!count) return fail(EGL_BAD_PARAMETER);
  const struct glr_config *match[NCONFIGS];
  EGLint n = 0;
  for (EGLint i = 0; i < NCONFIGS; i++) {
    const struct glr_config *k = &configs[i];
    int good = 1;
    for (const EGLint *p = attribs; p && p[0] != EGL_NONE && good; p += 2) {
      EGLint want = p[1];
      int known;
      EGLint have = config_attrib(k, p[0], &known);
      if (!known) {
        if (p[0] == EGL_MATCH_NATIVE_PIXMAP || p[0] == EGL_MAX_PBUFFER_WIDTH) continue;
        return fail(EGL_BAD_ATTRIBUTE);
      }
      if (want == EGL_DONT_CARE) continue;
      switch (p[0]) {
      case EGL_CONFIG_ID: good = have == want; break;
      case EGL_SURFACE_TYPE: good = !(want & (EGL_PBUFFER_BIT | EGL_WINDOW_BIT | EGL_PIXMAP_BIT) & ~have); break;
      case EGL_RENDERABLE_TYPE: case EGL_CONFORMANT: good = (want & have) == want; break;
      case EGL_CONFIG_CAVEAT: case EGL_COLOR_BUFFER_TYPE: case EGL_NATIVE_RENDERABLE: case EGL_TRANSPARENT_TYPE:
      case EGL_LEVEL: case EGL_NATIVE_VISUAL_TYPE: case EGL_BIND_TO_TEXTURE_RGB: case EGL_BIND_TO_TEXTURE_RGBA:
        good = have == want; break;
      case EGL_MAX_SWAP_INTERVAL: case EGL_MIN_SWAP_INTERVAL: case EGL_NATIVE_VISUAL_ID: case EGL_MAX_PBUFFER_HEIGHT:
      case EGL_MAX_PBUFFER_PIXELS: case EGL_TRANSPARENT_RED_VALUE: case EGL_TRANSPARENT_GREEN_VALUE: case EGL_TRANSPARENT_BLUE_VALUE:
        break;
      default: good = have >= want; break;
      }
    }
    if (good) match[n++] = k;
  }
  choose_request = attribs;
  qsort(match, (size_t)n, sizeof match[0], choose_compare);
  EGLint written = 0;
  if (!out) written = n;
  else
    for (; written < n && written < size; written++) out[written] = (EGLConfig)(intptr_t)match[written]->id;
  *count = written;
  return ok();
}
static EGLSurface eglCreatePbufferSurface(EGLDisplay dpy, EGLConfig config, const EGLint *attribs) {
  if (!valid_display(dpy, 1)) return EGL_NO_SURFACE;
  const struct glr_config *k = config_of(config);
  if (!k) { tls_error = EGL_BAD_CONFIG; return EGL_NO_SURFACE; }
  EGLint width = 0, height = 0;
  for (const EGLint *p = attribs; p && p[0] != EGL_NONE; p += 2) {
    switch (p[0]) {
    case EGL_WIDTH: width = p[1]; break;
    case EGL_HEIGHT: height = p[1]; break;
    case EGL_LARGEST_PBUFFER: case EGL_TEXTURE_FORMAT: case EGL_TEXTURE_TARGET: case EGL_MIPMAP_TEXTURE:
    case EGL_GL_COLORSPACE: case EGL_VG_ALPHA_FORMAT: case EGL_VG_COLORSPACE: break;
    default: tls_error = EGL_BAD_ATTRIBUTE; return EGL_NO_SURFACE;
    }
  }
  if (width < 0 || height < 0) { tls_error = EGL_BAD_PARAMETER; return EGL_NO_SURFACE; }
  struct glr_surface *s = calloc(1, sizeof *s);
  if (!s) { tls_error = EGL_BAD_ALLOC; return EGL_NO_SURFACE; }
  s->magic = SURFACE_MAGIC;
  s->width = width;
  s->height = height;
  s->config_id = k->id;
  pthread_mutex_lock(&egl_lock);
  s->next = surfaces;
  surfaces = s;
  pthread_mutex_unlock(&egl_lock);
  tls_error = EGL_SUCCESS;
  return (EGLSurface)s;
}
/* `attribs` holds EGLint or EGLAttrib pairs (wide). */
static EGLSurface create_window_surface(EGLDisplay dpy, EGLConfig config, unsigned long win, const void *attribs, int wide) {
  if (!valid_display(dpy, 1)) return EGL_NO_SURFACE;
  const struct glr_config *k = config_of(config);
  if (!k) { tls_error = EGL_BAD_CONFIG; return EGL_NO_SURFACE; }
  for (size_t i = 0; attribs; i += 2) {
    EGLAttrib name = wide ? ((const EGLAttrib *)attribs)[i] : ((const EGLint *)attribs)[i];
    EGLAttrib value = wide ? ((const EGLAttrib *)attribs)[i + 1] : ((const EGLint *)attribs)[i + 1];
    if (name == EGL_NONE) break;
    if (name == EGL_RENDER_BUFFER && value == EGL_BACK_BUFFER) continue;
    if (name == EGL_GL_COLORSPACE && value == EGL_GL_COLORSPACE_LINEAR) continue;
    if (name == EGL_RENDER_BUFFER || name == EGL_GL_COLORSPACE) { tls_error = EGL_BAD_MATCH; return EGL_NO_SURFACE; }
    tls_error = EGL_BAD_ATTRIBUTE;
    return EGL_NO_SURFACE;
  }
  int width, height;
  if (!window_ops || !win || !window_ops->size(win, &width, &height)) { tls_error = EGL_BAD_NATIVE_WINDOW; return EGL_NO_SURFACE; }
  struct glr_surface *s = calloc(1, sizeof *s);
  if (!s) { tls_error = EGL_BAD_ALLOC; return EGL_NO_SURFACE; }
  s->magic = SURFACE_MAGIC;
  s->width = width;
  s->height = height;
  s->config_id = k->id;
  s->window = 1;
  s->native = win;
  pthread_mutex_lock(&egl_lock);
  s->next = surfaces;
  surfaces = s;
  pthread_mutex_unlock(&egl_lock);
  if (tracing()) say("window surface 0x%lx %dx%d config %d", win, width, height, k->id);
  tls_error = EGL_SUCCESS;
  return (EGLSurface)s;
}
static EGLSurface eglCreateWindowSurface(EGLDisplay dpy, EGLConfig config, EGLNativeWindowType win, const EGLint *attribs) {
  return create_window_surface(dpy, config, (unsigned long)win, attribs, 0);
}
static EGLSurface eglCreatePlatformWindowSurface(EGLDisplay dpy, EGLConfig config, void *win, const EGLAttrib *attribs) {
  /* EGL_KHR_platform_x11: `win` points to the Window. */
  return create_window_surface(dpy, config, win ? *(const unsigned long *)win : 0, attribs, 1);
}
static EGLSurface eglCreatePlatformWindowSurfaceEXT(EGLDisplay dpy, EGLConfig config, void *win, const EGLint *attribs) {
  return create_window_surface(dpy, config, win ? *(const unsigned long *)win : 0, attribs, 0);
}
static EGLSurface eglCreatePixmapSurface(EGLDisplay dpy, EGLConfig config, EGLNativePixmapType pixmap, const EGLint *attribs) {
  (void)config; (void)pixmap; (void)attribs;
  if (!valid_display(dpy, 1)) return EGL_NO_SURFACE;
  tls_error = EGL_BAD_MATCH;
  return EGL_NO_SURFACE;
}
static EGLSurface eglCreatePlatformPixmapSurface(EGLDisplay dpy, EGLConfig config, void *pixmap, const EGLAttrib *attribs) {
  (void)config; (void)pixmap; (void)attribs;
  if (!valid_display(dpy, 1)) return EGL_NO_SURFACE;
  tls_error = EGL_BAD_MATCH;
  return EGL_NO_SURFACE;
}
static EGLSurface eglCreatePbufferFromClientBuffer(EGLDisplay dpy, EGLenum type, EGLClientBuffer buffer, EGLConfig config, const EGLint *attribs) {
  (void)type; (void)buffer; (void)config; (void)attribs;
  if (!valid_display(dpy, 1)) return EGL_NO_SURFACE;
  tls_error = EGL_BAD_PARAMETER;
  return EGL_NO_SURFACE;
}
static void unlink_surface(struct glr_surface *s) {
  for (struct glr_surface **p = &surfaces; *p; p = &(*p)->next)
    if (*p == s) { *p = s->next; break; }
  for (struct glr_ctx *c = contexts; c; c = c->next) {
    if (c->shown == s) c->shown = NULL;
    /* The key stays with the context; another of its windows can take it. */
    if (c->key_surface == s) c->key_surface = NULL;
  }
  s->magic = 0;
  free(s->pixels);
  free(s->shown_px);
  free(s);
}
static EGLBoolean eglDestroySurface(EGLDisplay dpy, EGLSurface surface) {
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  pthread_mutex_lock(&egl_lock);
  struct glr_surface *s = surface_of(surface);
  if (s) {
    if (s->bound) s->destroy_pending = 1;
    else unlink_surface(s);
  }
  pthread_mutex_unlock(&egl_lock);
  return s ? ok() : fail(EGL_BAD_SURFACE);
}
/* Sizes the current context's host canvas; a new size clears it. */
static void set_canvas(struct glr_ctx *c, EGLint w, EGLint h) {
  if (w <= 0) w = 1;
  if (h <= 0) h = 1;
  if (w != c->canvas_w || h != c->canvas_h) c->shown = NULL;
  c->canvas_w = w;
  c->canvas_h = h;
  CMD(SetSurface, (uint32_t)w, (uint32_t)h);
}
/* Window surfaces follow their window's size; the host canvas follows the
 * current draw surface. Returns 1 when the size changed. */
static int refresh_window(struct glr_surface *s) {
  int w, h;
  if (!s->window || !window_ops->size(s->native, &w, &h) || (w == s->width && h == s->height)) return 0;
  s->width = w;
  s->height = h;
  s->presented = 0;
  struct glr_ctx *c = tls_ctx;
  if (c && c->draw == s) set_canvas(c, w, h);
  return 1;
}
static EGLBoolean eglQuerySurface(EGLDisplay dpy, EGLSurface surface, EGLint attribute, EGLint *value) {
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  struct glr_surface *s = surface_of(surface);
  if (!s) return fail(EGL_BAD_SURFACE);
  if (s->window && (attribute == EGL_WIDTH || attribute == EGL_HEIGHT)) refresh_window(s);
  switch (attribute) {
  case EGL_WIDTH: *value = s->width; break;
  case EGL_HEIGHT: *value = s->height; break;
  case EGL_CONFIG_ID: *value = s->config_id; break;
  case EGL_LARGEST_PBUFFER: case EGL_MIPMAP_TEXTURE: case EGL_MIPMAP_LEVEL: *value = 0; break;
  case EGL_TEXTURE_FORMAT: case EGL_TEXTURE_TARGET: *value = EGL_NO_TEXTURE; break;
  case EGL_RENDER_BUFFER: *value = EGL_BACK_BUFFER; break;
  case EGL_SWAP_BEHAVIOR: *value = EGL_BUFFER_PRESERVED; break;
  case EGL_MULTISAMPLE_RESOLVE: *value = EGL_MULTISAMPLE_RESOLVE_DEFAULT; break;
  case EGL_HORIZONTAL_RESOLUTION: case EGL_VERTICAL_RESOLUTION: case EGL_PIXEL_ASPECT_RATIO: *value = EGL_UNKNOWN; break;
  case EGL_GL_COLORSPACE: *value = EGL_GL_COLORSPACE_LINEAR; break;
  case EGL_VG_ALPHA_FORMAT: *value = EGL_VG_ALPHA_FORMAT_NONPRE; break;
  case EGL_VG_COLORSPACE: *value = EGL_VG_COLORSPACE_sRGB; break;
  case EGL_BUFFER_AGE_EXT: {
    /* The canvas keeps its drawing buffer, so it still holds the last frame
     * swapped from this surface unless another surface or a resize used it. */
    struct glr_ctx *c = tls_ctx;
    if (!c || c->draw != s) return fail(EGL_BAD_SURFACE);
    *value = s->window && c->shown == s ? 1 : 0;
    c->prof.buffer_age_queries += *value;
    break;
  }
  default: return fail(EGL_BAD_ATTRIBUTE);
  }
  return ok();
}
static EGLBoolean eglSurfaceAttrib(EGLDisplay dpy, EGLSurface surface, EGLint attribute, EGLint value) {
  (void)attribute; (void)value;
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  return surface_of(surface) ? ok() : fail(EGL_BAD_SURFACE);
}
static EGLBoolean eglBindAPI(EGLenum api) {
  if (api != EGL_OPENGL_ES_API) return fail(EGL_BAD_PARAMETER);
  tls_api = api;
  return ok();
}
static EGLenum eglQueryAPI(void) { return tls_api; }

/* Host info text, see guest-gl-remote.js openInfo(). */
static int parse_open_info(struct glr_ctx *c, char *text) {
  c->info = text;
  size_t next_cap = 0;
  for (char *p = text; *p;) {
    char *nl = strchr(p, '\n');
    if (nl) *nl = 0;
    char *line = p;
    p = nl ? nl + 1 : p + strlen(p);
    char *value = strchr(line, ' ');
    if (!value) continue;
    *value++ = 0;
    if (!strcmp(line, "version")) c->version = value;
    else if (!strcmp(line, "renderer")) c->renderer = value;
    else if (!strcmp(line, "vendor")) c->vendor = value;
    else if (!strcmp(line, "glsl")) c->glsl = value;
    else if (!strcmp(line, "present")) c->keyed = !strcmp(value, "keyed");
    else if (!strcmp(line, "ext")) {
      c->extensions = strdup(value);
      for (char *e = value; *e;) {
        while (*e == ' ') e++;
        if (!*e) break;
        if ((size_t)c->n_ext == next_cap) {
          next_cap = next_cap ? next_cap * 2 : 32;
          const char **grown = realloc(c->ext, next_cap * sizeof *grown);
          if (!grown) return 0;
          c->ext = grown;
        }
        c->ext[c->n_ext++] = e;
        while (*e && *e != ' ') e++;
        if (*e) *e++ = 0;
      }
    } else if (!strcmp(line, "p")) {
      struct limit *grown = realloc(c->limits, (size_t)(c->nlimits + 1) * sizeof *grown);
      if (!grown) return 0;
      c->limits = grown;
      struct limit *l = &c->limits[c->nlimits];
      char *end;
      l->pname = (GLenum)strtoul(value, &end, 10);
      l->n = 0;
      l->v = NULL;
      for (char *q = end; *q;) {
        double x = strtod(q, &end);
        if (end == q) break;
        double *more = realloc(l->v, (size_t)(l->n + 1) * sizeof *more);
        if (!more) return 0;
        l->v = more;
        l->v[l->n++] = x;
        q = end;
      }
      c->nlimits++;
    }
  }
  if (!c->extensions) c->extensions = strdup("");
  return c->version && c->renderer && c->vendor && c->glsl && c->extensions;
}
static void free_context(struct glr_ctx *c) {
  if (c->host > 0) hostcall(HC_GL_CLOSE, c->host, 0, 0, 0, 0);
  c->cookie[0] = c->cookie[1] = 0;
  if (tracing()) say("context %d closed: execs=%llu words=%llu queries=%llu", c->host, (unsigned long long)c->execs,
                     (unsigned long long)c->words, (unsigned long long)c->queries);
  for (size_t i = 0; i < c->nobjs; i++)
    if (c->objs[i]) { clear_info(c->objs[i]); free(c->objs[i]->source); free(c->objs[i]); }
  for (int i = 0; i < c->nlimits; i++) free(c->limits[i].v);
  for (int i = 0; i < 8; i++) free(c->maps[i].ptr);
  for (size_t i = 0; i < c->nbufs; i++) free(c->bufs[i].shadow);
  for (size_t i = 0; i < c->nstaged; i++) free(c->staged[i]);
  for (int i = 0; i < 4; i++) free(c->spare[i]);
  free(c->bufs); free(c->staged);
  free(c->objs); free(c->limits); free(c->ext); free((void *)c->extensions); free(c->info);
  free(c->vao_elements); free(c->cmd); free(c->reply); free(c);
}
static EGLContext eglCreateContext(EGLDisplay dpy, EGLConfig config, EGLContext share, const EGLint *attribs) {
  if (!valid_display(dpy, 1)) return EGL_NO_CONTEXT;
  if (tls_api != EGL_OPENGL_ES_API) { tls_error = EGL_BAD_MATCH; return EGL_NO_CONTEXT; }
  const struct glr_config *k = NULL;
  if (config != EGL_NO_CONFIG_KHR && !(k = config_of(config))) { tls_error = EGL_BAD_CONFIG; return EGL_NO_CONTEXT; }
  EGLint major = 1, minor = 0;
  for (const EGLint *p = attribs; p && p[0] != EGL_NONE; p += 2) {
    switch (p[0]) {
    case EGL_CONTEXT_MAJOR_VERSION: major = p[1]; break;
    case EGL_CONTEXT_MINOR_VERSION: minor = p[1]; break;
    case EGL_CONTEXT_FLAGS_KHR: case EGL_CONTEXT_OPENGL_PROFILE_MASK: case EGL_CONTEXT_OPENGL_DEBUG:
    case EGL_CONTEXT_OPENGL_FORWARD_COMPATIBLE: case EGL_CONTEXT_OPENGL_ROBUST_ACCESS: case EGL_CONTEXT_OPENGL_ROBUST_ACCESS_EXT:
    case EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY: case EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY_EXT: break;
    default: tls_error = EGL_BAD_ATTRIBUTE; return EGL_NO_CONTEXT;
    }
  }
  if (major < 2 || major > 3 || (major == 3 && minor > 0) || (major == 2 && minor != 0)) { tls_error = EGL_BAD_MATCH; return EGL_NO_CONTEXT; }
  pthread_mutex_lock(&egl_lock);
  struct glr_ctx *shared = share == EGL_NO_CONTEXT ? NULL : context_of(share);
  pthread_mutex_unlock(&egl_lock);
  if (share != EGL_NO_CONTEXT && !shared) { tls_error = EGL_BAD_CONTEXT; return EGL_NO_CONTEXT; }
  if (shared) say("shared contexts are not supported; context %d objects stay private", shared->host);
  struct glr_ctx *c = calloc(1, sizeof *c);
  char *info = malloc(INFO_CAP + 1);
  if (!c || !info || !(c->cmd = malloc(CMD_WORDS * 4)) || !(c->reply = malloc(65536))) {
    free(info);
    if (c) { free(c->cmd); free(c); }
    tls_error = EGL_BAD_ALLOC;
    return EGL_NO_CONTEXT;
  }
  c->reply_cap = 65536;
  uint32_t flags = (!k || k->alpha ? OPEN_ALPHA : 0) | (!k || k->depth ? OPEN_DEPTH : 0) | (!k || k->stencil ? OPEN_STENCIL : 0);
  if (getrandom(c->cookie, sizeof c->cookie, GRND_NONBLOCK) != (ssize_t)sizeof c->cookie) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    c->cookie[0] = (uint32_t)t.tv_nsec ^ (uint32_t)(uintptr_t)c;
    c->cookie[1] = (uint32_t)t.tv_sec ^ (uint32_t)getpid() << 16;
  }
  long r = hostcall(HC_GL_OPEN, flags, (long)(uintptr_t)info, INFO_CAP, (long)(uintptr_t)c->cookie, 0);
  if (r <= 0) {
    say("host context open failed: %ld", r);
    free(info); free(c->cmd); free(c->reply); free(c);
    tls_error = EGL_BAD_ALLOC;
    return EGL_NO_CONTEXT;
  }
  c->host = (int32_t)r;
  /* OPEN wrote a NUL-terminated text of at most INFO_CAP bytes. */
  info[INFO_CAP] = 0;
  if (!parse_open_info(c, info)) {
    say("host context %d sent incomplete info", c->host);
    free_context(c);
    tls_error = EGL_BAD_ALLOC;
    return EGL_NO_CONTEXT;
  }
  c->config_id = k ? k->id : 0;
  if (software_renderer(c->renderer)) host_software = 1;
  c->ps[PS_UNPACK_ALIGNMENT] = c->ps[PS_PACK_ALIGNMENT] = 4;
  c->enabled[cap_index(GL_DITHER)] = 1;
  c->viewport[2] = c->viewport[3] = c->scissor[2] = c->scissor[3] = 1;
  pthread_mutex_lock(&egl_lock);
  c->next = contexts;
  contexts = c;
  pthread_mutex_unlock(&egl_lock);
  if (tracing()) say("context %d: %s | %s | %s | %d extensions", c->host, c->version, c->renderer, c->glsl, c->n_ext);
  tls_error = EGL_SUCCESS;
  return (EGLContext)c;
}
static void unlink_context(struct glr_ctx *c) {
  for (struct glr_ctx **p = &contexts; *p; p = &(*p)->next)
    if (*p == c) { *p = c->next; break; }
}
static EGLBoolean eglDestroyContext(EGLDisplay dpy, EGLContext ctx) {
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  pthread_mutex_lock(&egl_lock);
  struct glr_ctx *c = context_of(ctx);
  int now = c && !c->bound;
  if (c && c->bound) c->destroy_pending = 1;
  if (now) unlink_context(c);
  pthread_mutex_unlock(&egl_lock);
  if (!c) return fail(EGL_BAD_CONTEXT);
  if (now) free_context(c);
  return ok();
}
/* Releases the thread's current context; frees what was destroyed meanwhile. */
static void release_current(void) {
  struct glr_ctx *c = tls_ctx;
  if (!c) return;
  flush(c);
  tls_ctx = NULL;
  pthread_mutex_lock(&egl_lock);
  c->bound = 0;
  struct glr_surface *old[2] = {c->draw, c->read};
  for (int i = 0; i < 2; i++)
    if (old[i]) {
      old[i]->bound = 0;
      if (old[i]->destroy_pending && (i == 0 || old[1] != old[0])) unlink_surface(old[i]);
    }
  c->draw = c->read = NULL;
  int destroy = c->destroy_pending;
  if (destroy) unlink_context(c);
  pthread_mutex_unlock(&egl_lock);
  if (destroy) free_context(c);
}
static EGLBoolean eglMakeCurrent(EGLDisplay dpy, EGLSurface draw, EGLSurface read, EGLContext ctx) {
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  if (ctx == EGL_NO_CONTEXT) {
    if (draw != EGL_NO_SURFACE || read != EGL_NO_SURFACE) return fail(EGL_BAD_MATCH);
    release_current();
    return ok();
  }
  pthread_mutex_lock(&egl_lock);
  struct glr_ctx *c = context_of(ctx);
  struct glr_surface *d = draw == EGL_NO_SURFACE ? NULL : surface_of(draw);
  struct glr_surface *r = read == EGL_NO_SURFACE ? NULL : surface_of(read);
  EGLint error = !c || c->destroy_pending ? EGL_BAD_CONTEXT
               : (draw != EGL_NO_SURFACE && !d) || (read != EGL_NO_SURFACE && !r) ? EGL_BAD_SURFACE
               : (!d) != (!r) ? EGL_BAD_MATCH
               : c->bound && c != tls_ctx ? EGL_BAD_ACCESS : EGL_SUCCESS;
  pthread_mutex_unlock(&egl_lock);
  if (error != EGL_SUCCESS) return fail(error);
  if (c != tls_ctx) release_current();
  pthread_mutex_lock(&egl_lock);
  if (c->draw && c->draw != d) c->draw->bound = 0;
  if (c->read && c->read != r) c->read->bound = 0;
  c->bound = 1;
  c->draw = d;
  c->read = r;
  if (d) d->bound = 1;
  if (r) r->bound = 1;
  pthread_mutex_unlock(&egl_lock);
  tls_ctx = c;
  if (d && d->window) {
    int w, h;
    if (window_ops->size(d->native, &w, &h) && (w != d->width || h != d->height)) {
      d->width = w;
      d->height = h;
      d->presented = 0;
    }
  }
  if (d) {
    EGLint w = d->width > 0 ? d->width : 1, h = d->height > 0 ? d->height : 1;
    set_canvas(c, w, h);
    if (!c->made_current) {
      /* GL sets viewport and scissor to the surface the first time a context is made current with one. */
      glViewport(0, 0, w, h);
      glScissor(0, 0, w, h);
    }
    c->made_current = 1;
  }
  return ok();
}
static EGLContext eglGetCurrentContext(void) { return tls_ctx ? (EGLContext)tls_ctx : EGL_NO_CONTEXT; }
static EGLDisplay eglGetCurrentDisplay(void) { return tls_ctx ? DISPLAY : EGL_NO_DISPLAY; }
static EGLSurface eglGetCurrentSurface(EGLint which) {
  if (!tls_ctx) return EGL_NO_SURFACE;
  if (which == EGL_DRAW) return tls_ctx->draw ? (EGLSurface)tls_ctx->draw : EGL_NO_SURFACE;
  if (which == EGL_READ) return tls_ctx->read ? (EGLSurface)tls_ctx->read : EGL_NO_SURFACE;
  tls_error = EGL_BAD_PARAMETER;
  return EGL_NO_SURFACE;
}
static EGLBoolean eglQueryContext(EGLDisplay dpy, EGLContext ctx, EGLint attribute, EGLint *value) {
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  struct glr_ctx *c = context_of(ctx);
  if (!c) return fail(EGL_BAD_CONTEXT);
  switch (attribute) {
  case EGL_CONFIG_ID: *value = c->config_id; break;
  case EGL_CONTEXT_CLIENT_TYPE: *value = EGL_OPENGL_ES_API; break;
  case EGL_CONTEXT_CLIENT_VERSION: *value = 3; break;
  case EGL_RENDER_BUFFER: *value = c->draw ? EGL_BACK_BUFFER : EGL_NONE; break;
  default: return fail(EGL_BAD_ATTRIBUTE);
  }
  return ok();
}
static EGLint eglGetError(void) {
  EGLint e = tls_error;
  tls_error = EGL_SUCCESS;
  return e;
}
/* Index of the first word from the left (from_right: last word from the
 * right) where a and b differ in n words; -1 when they are equal. */
static int first_diff(const uint32_t *a, const uint32_t *b, int n, int from_right) {
  int i = 0;
#ifdef __wasm_simd128__
  if (!from_right) {
    for (; i + 16 <= n; i += 16) {
      v128_t d = wasm_v128_or(wasm_v128_or(wasm_v128_xor(wasm_v128_load(a + i), wasm_v128_load(b + i)),
                                           wasm_v128_xor(wasm_v128_load(a + i + 4), wasm_v128_load(b + i + 4))),
                              wasm_v128_or(wasm_v128_xor(wasm_v128_load(a + i + 8), wasm_v128_load(b + i + 8)),
                                           wasm_v128_xor(wasm_v128_load(a + i + 12), wasm_v128_load(b + i + 12))));
      if (wasm_v128_any_true(d)) break;
    }
  } else {
    for (; i + 4 <= n && !wasm_v128_any_true(wasm_v128_xor(wasm_v128_load(a + n - i - 4), wasm_v128_load(b + n - i - 4))); i += 4) {}
  }
#endif
  for (; i < n; i++)
    if (from_right ? a[n - 1 - i] != b[n - 1 - i] : a[i] != b[i]) return from_right ? n - 1 - i : i;
  return -1;
}
/* Narrows the w x h frame rectangle at x,top (top-left origin) of `px` to the
 * box that differs from what the window shows; returns 0 when nothing does.
 * The box is moved to the start of `px`, rows `*bw` words apart. */
static int changed_box(struct glr_surface *s, uint32_t *px, int x, int top, int w, int h, int *bx, int *by, int *bw, int *bh) {
  int r0 = -1, r1 = -1, c0 = w, c1 = -1;
  for (int r = 0; r < h; r++) {
    const uint32_t *now = px + (size_t)r * (size_t)w, *old = s->shown_px + (size_t)(top + r) * (size_t)s->shown_w + x;
    int a = first_diff(now, old, w, 0);
    if (a < 0) continue;
    if (r0 < 0) r0 = r;
    r1 = r;
    int b = first_diff(now, old, w, 1);
    if (a < c0) c0 = a;
    if (b > c1) c1 = b;
  }
  if (r0 < 0) return 0;
  *bx = x + c0;
  *by = top + r0;
  *bw = c1 - c0 + 1;
  *bh = r1 - r0 + 1;
  for (int r = r0; r <= r1; r++)
    memmove(px + (size_t)(r - r0) * (size_t)*bw, px + (size_t)r * (size_t)w + c0, (size_t)*bw * 4);
  return 1;
}

/* Reads the rectangle x,y,w,h (GL coordinates, bottom-left origin) of the
 * host canvas (bottom row first, RGBA bytes) and presents it top row first as
 * 0xAARRGGBB words. The context's pack state is restored. Unless `whole`,
 * only the part that differs from what the window shows is sent. */
static int present_window(struct glr_ctx *c, struct glr_surface *s, int x, int y, int w, int h, int whole) {
  if (w <= 0 || h <= 0) return 1;
  size_t words = (size_t)w * (size_t)h;
  if (words > s->pixels_cap) {
    uint32_t *grown = realloc(s->pixels, words * 4);
    if (!grown) return 0;
    s->pixels = grown;
    s->pixels_cap = words;
  }
  GLuint read_fb = c->read_fb, pack = c->pack_buffer;
  GLint ps[4] = {c->ps[PS_PACK_ALIGNMENT], c->ps[PS_PACK_ROW_LENGTH], c->ps[PS_PACK_SKIP_PIXELS], c->ps[PS_PACK_SKIP_ROWS]};
  static const GLenum ps_pack[4] = {GL_PACK_ALIGNMENT, GL_PACK_ROW_LENGTH, GL_PACK_SKIP_PIXELS, GL_PACK_SKIP_ROWS};
  static const GLint ps_default[4] = {4, 0, 0, 0};
  if (read_fb) glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
  if (pack) glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
  for (int i = 0; i < 4; i++)
    if (ps[i] != ps_default[i]) glPixelStorei(ps_pack[i], ps_default[i]);
  glReadPixels(x, y, w, h, GL_RGBA, GL_UNSIGNED_BYTE, s->pixels);
  for (int i = 0; i < 4; i++)
    if (ps[i] != ps_default[i]) glPixelStorei(ps_pack[i], ps[i]);
  if (pack) glBindBuffer(GL_PIXEL_PACK_BUFFER, pack);
  if (read_fb) glBindFramebuffer(GL_READ_FRAMEBUFFER, read_fb);
  if (c->lost) return 0;
  /* Flip rows and swap R and B: little-endian RGBA bytes are 0xAABBGGRR. */
  for (int top = 0, bottom = h - 1; top <= bottom; top++, bottom--) {
    uint32_t *a = s->pixels + (size_t)top * (size_t)w, *b = s->pixels + (size_t)bottom * (size_t)w;
    int x = 0;
#ifdef __wasm_simd128__
    for (; x + 4 <= w; x += 4) {
      v128_t va = wasm_v128_load(a + x), vb = wasm_v128_load(b + x);
      wasm_v128_store(a + x, wasm_i8x16_shuffle(vb, vb, 2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15));
      if (a != b) wasm_v128_store(b + x, wasm_i8x16_shuffle(va, va, 2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15));
    }
#endif
    for (; x < w; x++) {
      uint32_t va = a[x], vb = b[x];
      a[x] = (vb & 0xff00ff00u) | (vb & 0xffu) << 16 | (vb >> 16 & 0xffu);
      if (a != b) b[x] = (va & 0xff00ff00u) | (va & 0xffu) << 16 | (va >> 16 & 0xffu);
    }
  }
  int top = s->height - y - h;
  int known = s->shown_px && s->shown_w == s->width && s->shown_h == s->height;
  if (whole || !known) {
    /* The window's contents are unknown: send all, and remember it if it is the whole frame. */
    if (!known && (w != s->width || h != s->height)) {
      s->shown_w = s->shown_h = 0;
    } else {
      size_t all = (size_t)s->width * (size_t)s->height;
      if (!known) {
        free(s->shown_px);
        s->shown_px = malloc(all * 4);
        s->shown_w = s->shown_px ? s->width : 0;
        s->shown_h = s->shown_px ? s->height : 0;
      }
      for (int r = 0; s->shown_px && r < h; r++)
        memcpy(s->shown_px + (size_t)(top + r) * (size_t)s->width + x, s->pixels + (size_t)r * (size_t)w, (size_t)w * 4);
    }
  } else {
    int bx, by, bw, bh;
    if (!changed_box(s, s->pixels, x, top, w, h, &bx, &by, &bw, &bh)) return 1;
    for (int k = 0; k < bh; k++)
      memcpy(s->shown_px + (size_t)(by + k) * (size_t)s->width + bx, s->pixels + (size_t)k * (size_t)bw, (size_t)bw * 4);
    x = bx; top = by; w = bw; h = bh;
  }
  uint64_t t0 = tracing() >= 2 ? now_ns() : 0;
  int shown = window_ops->present(s->native, s->pixels, x, top, w, h);
  if (!shown) s->shown_w = s->shown_h = 0;
  if (t0) {
    c->prof.window_ns += now_ns() - t0;
    c->prof.swap_pixels += (uint64_t)w * (uint64_t)h;
  }
  return shown;
}
/* Keyed presentation of the damage x0,y0-x1,y1 (bottom-left origin): the
 * host posts the canvas to the page, which shows it wherever the window shows
 * the context's key colour. Returns 1 when presented so, 0 when the swap has
 * to read back instead (the window then has to be sent whole). */
static int present_keyed(struct glr_ctx *c, struct glr_surface *s, int x0, int y0, int x1, int y1, int whole) {
  if (!c->keyed || !window_ops->origin || !window_ops->fill) return 0;
  if (c->key_surface && c->key_surface != s) return 0;
  if (s->key_wait > 0) { s->key_wait--; return 0; }
  int rx, ry;
  if (!window_ops->origin(s->native, &rx, &ry)) return 0;
  if (!s->key || whole) { x0 = 0; y0 = 0; x1 = s->width; y1 = s->height; }
  uint32_t start;
  uint32_t *w = emit_query(c, OP_PresentSurface, 7, &start);
  w[0] = 0;
  w[1] = (uint32_t)rx;
  w[2] = (uint32_t)ry;
  w[3] = (uint32_t)x0;
  w[4] = (uint32_t)(s->height - y1);
  w[5] = (uint32_t)(x1 - x0);
  w[6] = (uint32_t)(y1 - y0);
  double key = 0;
  if (query_numbers(c, start, &key, 1) < 1 || key <= 0 || key > 0xffffff) {
    /* No keyed presenter on the page, or all keys taken. */
    if (c->key_surface == s) c->key_surface = NULL;
    s->key = 0;
    s->key_wait = 60;
    return 0;
  }
  c->key_surface = s;
  /* The frame is on the page before the window shows the key, so the key is
   * never shown without it. */
  if (s->key != (unsigned)key || whole) {
    if (!window_ops->fill(s->native, (unsigned)key)) {
      uint32_t end;
      uint32_t *e = emit_query(c, OP_PresentSurface, 7, &end);
      memset(e, 0, 7 * 4);
      e[0] = 1; /* end */
      query_numbers(c, end, &key, 1);
      c->key_surface = NULL;
      s->key = 0;
      s->key_wait = 600;
      return 0;
    }
    s->key = (unsigned)key;
  }
  return 1;
}
/* rects: n x,y,w,h damage rectangles (bottom-left origin); n == 0 means
 * the whole surface. Only their bounding box is read back, and only the part
 * of it that changed is presented, unless the window has not shown a whole
 * frame of this size yet or the window system reports that its contents were
 * lost. */
static EGLBoolean swap_buffers(EGLDisplay dpy, EGLSurface surface, const EGLint *rects, EGLint n) {
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  struct glr_surface *s = surface_of(surface);
  if (!s) return fail(EGL_BAD_SURFACE);
  if (n < 0 || (n > 0 && !rects)) return fail(EGL_BAD_PARAMETER);
  struct glr_ctx *c = tls_ctx;
  if (!c || c->draw != s) return s->window ? fail(EGL_BAD_SURFACE) : ok();
  if (!s->window) {
    flush(c);
    return ok();
  }
  uint64_t t0 = tracing() >= 2 ? now_ns() : 0;
  int x0 = 0, y0 = 0, x1 = s->width, y1 = s->height;
  int exposed = window_ops->exposed && window_ops->exposed(s->native);
  c->prof.swap_damaged += n > 0;
  if (tracing() >= 2 && c->prof.swaps % 50 == 0) {
    char text[256];
    int used = 0;
    for (EGLint i = 0; i < n && i < 6 && used < 200; i++)
      used += snprintf(text + used, sizeof text - (size_t)used, " %d,%d %dx%d", rects[4 * i], rects[4 * i + 1], rects[4 * i + 2], rects[4 * i + 3]);
    text[used] = 0;
    say("swap ctx=%d surface %dx%d damage n=%d:%s", c->host, s->width, s->height, n, text);
  }
  c->prof.swap_exposed += exposed;
  c->prof.swap_first += !s->presented;
  if (n > 0 && s->presented && !exposed) {
    x0 = s->width; y0 = s->height; x1 = 0; y1 = 0;
    for (EGLint i = 0; i < n; i++) {
      const EGLint *r = rects + 4 * i;
      int rx0 = r[0] < 0 ? 0 : r[0], ry0 = r[1] < 0 ? 0 : r[1];
      int rx1 = r[0] + r[2] > s->width ? s->width : r[0] + r[2];
      int ry1 = r[1] + r[3] > s->height ? s->height : r[1] + r[3];
      if (rx0 >= rx1 || ry0 >= ry1) continue;
      if (rx0 < x0) x0 = rx0;
      if (ry0 < y0) y0 = ry0;
      if (rx1 > x1) x1 = rx1;
      if (ry1 > y1) y1 = ry1;
    }
  }
  int whole = !s->presented || exposed;
  int shown = 1;
  unsigned had_key = s->key;
  if (x0 < x1 && y0 < y1 && !present_keyed(c, s, x0, y0, x1, y1, whole)) {
    if (had_key) {
      /* The window shows the key colour, not a frame. */
      s->key = 0;
      whole = 1;
      x0 = 0; y0 = 0; x1 = s->width; y1 = s->height;
    }
    shown = present_window(c, s, x0, y0, x1 - x0, y1 - y0, whole);
  }
  if (shown && !c->lost) {
    c->shown = s;
    s->presented = 1;
  }
  refresh_window(s);
  flush(c);
  if (t0) {
    c->prof.swaps++;
    c->prof.swap_ns += now_ns() - t0;
  }
  if (c->lost) return fail(EGL_CONTEXT_LOST);
  return shown ? ok() : fail(EGL_BAD_NATIVE_WINDOW);
}
static EGLBoolean eglSwapBuffers(EGLDisplay dpy, EGLSurface surface) {
  return swap_buffers(dpy, surface, NULL, 0);
}
static EGLBoolean eglSwapBuffersWithDamageKHR(EGLDisplay dpy, EGLSurface surface, const EGLint *rects, EGLint n) {
  return swap_buffers(dpy, surface, rects, n);
}
static EGLBoolean eglCopyBuffers(EGLDisplay dpy, EGLSurface surface, EGLNativePixmapType target) {
  (void)surface; (void)target;
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  return fail(EGL_BAD_NATIVE_PIXMAP);
}
static EGLBoolean eglSwapInterval(EGLDisplay dpy, EGLint interval) {
  (void)interval;
  return valid_display(dpy, 1) ? ok() : EGL_FALSE;
}
static EGLBoolean eglBindTexImage(EGLDisplay dpy, EGLSurface surface, EGLint buffer) {
  (void)surface; (void)buffer;
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  return fail(EGL_BAD_MATCH);
}
static EGLBoolean eglReleaseTexImage(EGLDisplay dpy, EGLSurface surface, EGLint buffer) {
  (void)surface; (void)buffer;
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  return fail(EGL_BAD_MATCH);
}
static EGLBoolean eglWaitGL(void) {
  if (tls_ctx) flush(tls_ctx);
  return ok();
}
static EGLBoolean eglWaitClient(void) { return eglWaitGL(); }
static EGLBoolean eglWaitNative(EGLint engine) {
  (void)engine;
  return ok();
}
static EGLBoolean eglReleaseThread(void) {
  release_current();
  tls_api = EGL_OPENGL_ES_API;
  return ok();
}
struct egl_sync { uint32_t magic; };
#define EGL_SYNC_MAGIC 0x65676c73u
static EGLSync eglCreateSync(EGLDisplay dpy, EGLenum type, const EGLAttrib *attribs) {
  (void)attribs;
  if (!valid_display(dpy, 1)) return EGL_NO_SYNC;
  if (type != EGL_SYNC_FENCE) { tls_error = EGL_BAD_ATTRIBUTE; return EGL_NO_SYNC; }
  if (!tls_ctx) { tls_error = EGL_BAD_MATCH; return EGL_NO_SYNC; }
  struct egl_sync *s = malloc(sizeof *s);
  if (!s) { tls_error = EGL_BAD_ALLOC; return EGL_NO_SYNC; }
  s->magic = EGL_SYNC_MAGIC;
  flush(tls_ctx);
  tls_error = EGL_SUCCESS;
  return (EGLSync)s;
}
static EGLBoolean eglDestroySync(EGLDisplay dpy, EGLSync sync) {
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  struct egl_sync *s = sync;
  if (!s || s->magic != EGL_SYNC_MAGIC) return fail(EGL_BAD_PARAMETER);
  s->magic = 0;
  free(s);
  return ok();
}
static EGLint eglClientWaitSync(EGLDisplay dpy, EGLSync sync, EGLint flags, EGLTime timeout) {
  (void)flags; (void)timeout;
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  struct egl_sync *s = sync;
  if (!s || s->magic != EGL_SYNC_MAGIC) return fail(EGL_BAD_PARAMETER);
  tls_error = EGL_SUCCESS;
  return EGL_CONDITION_SATISFIED;
}
static EGLBoolean eglGetSyncAttrib(EGLDisplay dpy, EGLSync sync, EGLint attribute, EGLAttrib *value) {
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  struct egl_sync *s = sync;
  if (!s || s->magic != EGL_SYNC_MAGIC) return fail(EGL_BAD_PARAMETER);
  switch (attribute) {
  case EGL_SYNC_TYPE: *value = EGL_SYNC_FENCE; break;
  case EGL_SYNC_STATUS: *value = EGL_SIGNALED; break;
  case EGL_SYNC_CONDITION: *value = EGL_SYNC_PRIOR_COMMANDS_COMPLETE; break;
  default: return fail(EGL_BAD_ATTRIBUTE);
  }
  return ok();
}
static EGLBoolean eglWaitSync(EGLDisplay dpy, EGLSync sync, EGLint flags) {
  (void)flags;
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  struct egl_sync *s = sync;
  return s && s->magic == EGL_SYNC_MAGIC ? ok() : fail(EGL_BAD_PARAMETER);
}
static EGLImage eglCreateImage(EGLDisplay dpy, EGLContext ctx, EGLenum target, EGLClientBuffer buffer, const EGLAttrib *attribs) {
  (void)ctx; (void)target; (void)buffer; (void)attribs;
  if (!valid_display(dpy, 1)) return EGL_NO_IMAGE;
  tls_error = EGL_BAD_PARAMETER;
  return EGL_NO_IMAGE;
}
static EGLBoolean eglDestroyImage(EGLDisplay dpy, EGLImage image) {
  (void)image;
  if (!valid_display(dpy, 1)) return EGL_FALSE;
  return fail(EGL_BAD_PARAMETER);
}
static __eglMustCastToProperFunctionPointerType eglGetProcAddress(const char *name);

/* ---- entry point tables ---- */

#define F(fn) {#fn, (glremote_proc)fn},
const struct glremote_export glremote_egl_table[] = {
  F(eglChooseConfig) F(eglCopyBuffers) F(eglCreateContext) F(eglCreatePbufferSurface) F(eglCreatePixmapSurface)
  F(eglCreateWindowSurface) F(eglDestroyContext) F(eglDestroySurface) F(eglGetConfigAttrib) F(eglGetConfigs)
  F(eglGetCurrentDisplay) F(eglGetCurrentSurface) F(eglGetDisplay) F(eglGetError) F(eglGetProcAddress)
  F(eglInitialize) F(eglMakeCurrent) F(eglQueryContext) F(eglQueryString) F(eglQuerySurface) F(eglSwapBuffers)
  F(eglTerminate) F(eglWaitGL) F(eglWaitNative) F(eglBindTexImage) F(eglReleaseTexImage) F(eglSurfaceAttrib)
  F(eglSwapInterval) F(eglBindAPI) F(eglQueryAPI) F(eglCreatePbufferFromClientBuffer) F(eglReleaseThread)
  F(eglWaitClient) F(eglGetCurrentContext) F(eglCreateSync) F(eglDestroySync) F(eglClientWaitSync)
  F(eglGetSyncAttrib) F(eglCreateImage) F(eglDestroyImage) F(eglGetPlatformDisplay) F(eglCreatePlatformWindowSurface)
  F(eglCreatePlatformPixmapSurface) F(eglWaitSync)
  {NULL, NULL},
};
static const struct glremote_export extra_table[] = {
  F(eglGetPlatformDisplayEXT) F(eglCreatePlatformWindowSurfaceEXT) F(eglQueryDisplayAttribEXT)
  F(eglQueryDeviceStringEXT) F(eglQueryDeviceAttribEXT) F(eglSwapBuffersWithDamageKHR)
  {"eglSwapBuffersWithDamageEXT", (glremote_proc)eglSwapBuffersWithDamageKHR},
  {NULL, NULL},
};
static const struct glremote_export gl_table[] = {
  F(glActiveTexture) F(glAttachShader) F(glBindAttribLocation) F(glBindBuffer) F(glBindFramebuffer) F(glBindRenderbuffer)
  F(glBindTexture) F(glBlendColor) F(glBlendEquation) F(glBlendEquationSeparate) F(glBlendFunc) F(glBlendFuncSeparate)
  F(glBufferData) F(glBufferSubData) F(glCheckFramebufferStatus) F(glClear) F(glClearColor) F(glClearDepthf)
  F(glClearStencil) F(glColorMask) F(glCompileShader) F(glCompressedTexImage2D) F(glCompressedTexSubImage2D)
  F(glCopyTexImage2D) F(glCopyTexSubImage2D) F(glCreateProgram) F(glCreateShader) F(glCullFace) F(glDeleteBuffers)
  F(glDeleteFramebuffers) F(glDeleteProgram) F(glDeleteRenderbuffers) F(glDeleteShader) F(glDeleteTextures)
  F(glDepthFunc) F(glDepthMask) F(glDepthRangef) F(glDetachShader) F(glDisable) F(glDisableVertexAttribArray)
  F(glDrawArrays) F(glDrawElements) F(glEnable) F(glEnableVertexAttribArray) F(glFinish) F(glFlush)
  F(glFramebufferRenderbuffer) F(glFramebufferTexture2D) F(glFrontFace) F(glGenBuffers) F(glGenerateMipmap)
  F(glGenFramebuffers) F(glGenRenderbuffers) F(glGenTextures) F(glGetActiveAttrib) F(glGetActiveUniform)
  F(glGetAttachedShaders) F(glGetAttribLocation) F(glGetBooleanv) F(glGetBufferParameteriv) F(glGetError)
  F(glGetFloatv) F(glGetFramebufferAttachmentParameteriv) F(glGetIntegerv) F(glGetProgramiv) F(glGetProgramInfoLog)
  F(glGetRenderbufferParameteriv) F(glGetShaderiv) F(glGetShaderInfoLog) F(glGetShaderPrecisionFormat)
  F(glGetShaderSource) F(glGetString) F(glGetTexParameterfv) F(glGetTexParameteriv) F(glGetUniformfv)
  F(glGetUniformiv) F(glGetUniformLocation) F(glGetVertexAttribfv) F(glGetVertexAttribiv) F(glGetVertexAttribPointerv)
  F(glHint) F(glIsBuffer) F(glIsEnabled) F(glIsFramebuffer) F(glIsProgram) F(glIsRenderbuffer) F(glIsShader)
  F(glIsTexture) F(glLineWidth) F(glLinkProgram) F(glPixelStorei) F(glPolygonOffset) F(glReadPixels)
  F(glReleaseShaderCompiler) F(glRenderbufferStorage) F(glSampleCoverage) F(glScissor) F(glShaderBinary)
  F(glShaderSource) F(glStencilFunc) F(glStencilFuncSeparate) F(glStencilMask) F(glStencilMaskSeparate)
  F(glStencilOp) F(glStencilOpSeparate) F(glTexImage2D) F(glTexParameterf) F(glTexParameterfv) F(glTexParameteri)
  F(glTexParameteriv) F(glTexSubImage2D) F(glUniform1f) F(glUniform1fv) F(glUniform1i) F(glUniform1iv)
  F(glUniform2f) F(glUniform2fv) F(glUniform2i) F(glUniform2iv) F(glUniform3f) F(glUniform3fv) F(glUniform3i)
  F(glUniform3iv) F(glUniform4f) F(glUniform4fv) F(glUniform4i) F(glUniform4iv) F(glUniformMatrix2fv)
  F(glUniformMatrix3fv) F(glUniformMatrix4fv) F(glUseProgram) F(glValidateProgram) F(glVertexAttrib1f)
  F(glVertexAttrib1fv) F(glVertexAttrib2f) F(glVertexAttrib2fv) F(glVertexAttrib3f) F(glVertexAttrib3fv)
  F(glVertexAttrib4f) F(glVertexAttrib4fv) F(glVertexAttribPointer) F(glViewport)
  F(glReadBuffer) F(glDrawRangeElements) F(glTexImage3D) F(glTexSubImage3D) F(glCopyTexSubImage3D)
  F(glCompressedTexImage3D) F(glCompressedTexSubImage3D) F(glGenQueries) F(glDeleteQueries) F(glIsQuery)
  F(glBeginQuery) F(glEndQuery) F(glGetQueryiv) F(glGetQueryObjectuiv) F(glUnmapBuffer) F(glGetBufferPointerv)
  F(glDrawBuffers) F(glUniformMatrix2x3fv) F(glUniformMatrix3x2fv) F(glUniformMatrix2x4fv) F(glUniformMatrix4x2fv)
  F(glUniformMatrix3x4fv) F(glUniformMatrix4x3fv) F(glBlitFramebuffer) F(glRenderbufferStorageMultisample)
  F(glFramebufferTextureLayer) F(glMapBufferRange) F(glFlushMappedBufferRange) F(glBindVertexArray)
  F(glDeleteVertexArrays) F(glGenVertexArrays) F(glIsVertexArray) F(glGetIntegeri_v) F(glBeginTransformFeedback)
  F(glEndTransformFeedback) F(glBindBufferRange) F(glBindBufferBase) F(glTransformFeedbackVaryings)
  F(glGetTransformFeedbackVarying) F(glVertexAttribIPointer) F(glGetVertexAttribIiv) F(glGetVertexAttribIuiv)
  F(glVertexAttribI4i) F(glVertexAttribI4ui) F(glVertexAttribI4iv) F(glVertexAttribI4uiv) F(glGetUniformuiv)
  F(glGetFragDataLocation) F(glUniform1ui) F(glUniform2ui) F(glUniform3ui) F(glUniform4ui) F(glUniform1uiv)
  F(glUniform2uiv) F(glUniform3uiv) F(glUniform4uiv) F(glClearBufferiv) F(glClearBufferuiv) F(glClearBufferfv)
  F(glClearBufferfi) F(glGetStringi) F(glCopyBufferSubData) F(glGetUniformIndices) F(glGetActiveUniformsiv)
  F(glGetUniformBlockIndex) F(glGetActiveUniformBlockiv) F(glGetActiveUniformBlockName) F(glUniformBlockBinding)
  F(glDrawArraysInstanced) F(glDrawElementsInstanced) F(glFenceSync) F(glIsSync) F(glDeleteSync)
  F(glClientWaitSync) F(glWaitSync) F(glGetInteger64v) F(glGetSynciv) F(glGetInteger64i_v)
  F(glGetBufferParameteri64v) F(glGenSamplers) F(glDeleteSamplers) F(glIsSampler) F(glBindSampler)
  F(glSamplerParameteri) F(glSamplerParameteriv) F(glSamplerParameterf) F(glSamplerParameterfv)
  F(glGetSamplerParameteriv) F(glGetSamplerParameterfv) F(glVertexAttribDivisor) F(glBindTransformFeedback)
  F(glDeleteTransformFeedbacks) F(glGenTransformFeedbacks) F(glIsTransformFeedback) F(glPauseTransformFeedback)
  F(glResumeTransformFeedback) F(glGetProgramBinary) F(glProgramBinary) F(glProgramParameteri)
  F(glInvalidateFramebuffer) F(glInvalidateSubFramebuffer) F(glTexStorage2D) F(glTexStorage3D) F(glGetInternalformativ)
  {NULL, NULL},
};
#undef F

glremote_proc glremote_get_proc_address(const char *name) {
  if (!name) return NULL;
  const struct glremote_export *tables[] = {gl_table, glremote_egl_table, extra_table};
  for (size_t t = 0; t < 3; t++)
    for (const struct glremote_export *e = tables[t]; e->name; e++)
      if (!strcmp(e->name, name)) return e->fn;
  if (tracing() >= 2) say("no entry point %s", name);
  return NULL;
}
static __eglMustCastToProperFunctionPointerType eglGetProcAddress(const char *name) {
  return (__eglMustCastToProperFunctionPointerType)glremote_get_proc_address(name);
}
