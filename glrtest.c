/* glrtest: guest check of glremote (GLES 3.0 on the host browser's WebGL2).
 * Creates a surfaceless-display ES 3.0 context with a pbuffer, reports the
 * identity strings, then verifies rendering by reading pixels back: clear,
 * a shader triangle with a uniform block and a webgl_-prefixed uniform, a
 * texelFetch from an uploaded texture, an FBO blit, and transform feedback.
 * Finally it times batched draws and query roundtrips. Lines start with
 * GLRTEST and go to stdout and /dev/console. */
#define _GNU_SOURCE
#define GL_GLES_PROTOTYPES 0
#define EGL_EGL_PROTOTYPES 0
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "glremote.h"

static int console = -1;
static void say(const char *format, ...) {
  char line[4096]; va_list args; va_start(args, format);
  int n = vsnprintf(line, sizeof line, format, args); va_end(args);
  if (n < 0) return; if (n >= (int)sizeof line) n = sizeof line - 1;
  fwrite(line, 1, n, stdout); fflush(stdout);
  if (console >= 0) write(console, line, n);
}

#define FUNCS(X) \
  X(EGLDisplay, eglGetPlatformDisplayEXT, (EGLenum, void *, const EGLint *)) \
  X(EGLBoolean, eglInitialize, (EGLDisplay, EGLint *, EGLint *)) X(EGLBoolean, eglBindAPI, (EGLenum)) \
  X(EGLBoolean, eglChooseConfig, (EGLDisplay, const EGLint *, EGLConfig *, EGLint, EGLint *)) \
  X(EGLContext, eglCreateContext, (EGLDisplay, EGLConfig, EGLContext, const EGLint *)) \
  X(EGLSurface, eglCreatePbufferSurface, (EGLDisplay, EGLConfig, const EGLint *)) \
  X(EGLBoolean, eglMakeCurrent, (EGLDisplay, EGLSurface, EGLSurface, EGLContext)) X(EGLint, eglGetError, (void)) \
  X(const char *, eglQueryString, (EGLDisplay, EGLint)) X(EGLBoolean, eglDestroyContext, (EGLDisplay, EGLContext)) \
  X(EGLBoolean, eglTerminate, (EGLDisplay)) \
  X(const GLubyte *, glGetString, (GLenum)) X(void, glGetIntegerv, (GLenum, GLint *)) X(GLenum, glGetError, (void)) \
  X(void, glClearColor, (GLfloat, GLfloat, GLfloat, GLfloat)) X(void, glClear, (GLbitfield)) \
  X(void, glReadPixels, (GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void *)) X(void, glViewport, (GLint, GLint, GLsizei, GLsizei)) \
  X(GLuint, glCreateShader, (GLenum)) X(void, glShaderSource, (GLuint, GLsizei, const GLchar *const *, const GLint *)) \
  X(void, glCompileShader, (GLuint)) X(void, glGetShaderiv, (GLuint, GLenum, GLint *)) \
  X(void, glGetShaderInfoLog, (GLuint, GLsizei, GLsizei *, GLchar *)) X(GLuint, glCreateProgram, (void)) \
  X(void, glAttachShader, (GLuint, GLuint)) X(void, glLinkProgram, (GLuint)) X(void, glGetProgramiv, (GLuint, GLenum, GLint *)) \
  X(void, glGetProgramInfoLog, (GLuint, GLsizei, GLsizei *, GLchar *)) X(void, glUseProgram, (GLuint)) \
  X(GLint, glGetUniformLocation, (GLuint, const GLchar *)) X(GLint, glGetAttribLocation, (GLuint, const GLchar *)) \
  X(GLuint, glGetUniformBlockIndex, (GLuint, const GLchar *)) X(void, glUniformBlockBinding, (GLuint, GLuint, GLuint)) \
  X(void, glUniform4f, (GLint, GLfloat, GLfloat, GLfloat, GLfloat)) X(void, glUniform1i, (GLint, GLint)) \
  X(void, glGenBuffers, (GLsizei, GLuint *)) X(void, glBindBuffer, (GLenum, GLuint)) \
  X(void, glBufferData, (GLenum, GLsizeiptr, const void *, GLenum)) X(void, glBindBufferBase, (GLenum, GLuint, GLuint)) \
  X(void, glGenVertexArrays, (GLsizei, GLuint *)) X(void, glBindVertexArray, (GLuint)) \
  X(void, glEnableVertexAttribArray, (GLuint)) X(void, glVertexAttribPointer, (GLuint, GLint, GLenum, GLboolean, GLsizei, const void *)) \
  X(void, glDrawArrays, (GLenum, GLint, GLsizei)) X(void, glGenTextures, (GLsizei, GLuint *)) X(void, glBindTexture, (GLenum, GLuint)) \
  X(void, glTexImage2D, (GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void *)) \
  X(void, glTexStorage2D, (GLenum, GLsizei, GLenum, GLsizei, GLsizei)) X(void, glTexParameteri, (GLenum, GLenum, GLint)) \
  X(void, glGenFramebuffers, (GLsizei, GLuint *)) X(void, glBindFramebuffer, (GLenum, GLuint)) \
  X(void, glFramebufferTexture2D, (GLenum, GLenum, GLenum, GLuint, GLint)) X(GLenum, glCheckFramebufferStatus, (GLenum)) \
  X(void, glBlitFramebuffer, (GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum)) \
  X(void, glTransformFeedbackVaryings, (GLuint, GLsizei, const GLchar *const *, GLenum)) \
  X(void, glBeginTransformFeedback, (GLenum)) X(void, glEndTransformFeedback, (void)) X(void, glEnable, (GLenum)) \
  X(void, glDisable, (GLenum)) X(void *, glMapBufferRange, (GLenum, GLintptr, GLsizeiptr, GLbitfield)) \
  X(GLboolean, glUnmapBuffer, (GLenum)) X(void, glFinish, (void)) \
  X(void, glDeleteTextures, (GLsizei, const GLuint *)) X(void, glDeleteFramebuffers, (GLsizei, const GLuint *))
#define DECLARE(ret, name, args) static ret (*name) args;
FUNCS(DECLARE)

static int failures;
static void check(int ok, const char *what) {
  say("GLRTEST %s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) failures++;
}
static double now_ms(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
static int near(const unsigned char *p, int r, int g, int b, int a) {
  return abs(p[0] - r) <= 2 && abs(p[1] - g) <= 2 && abs(p[2] - b) <= 2 && abs(p[3] - a) <= 2;
}

static GLuint shader(GLenum type, const char *source) {
  GLuint s = glCreateShader(type); glShaderSource(s, 1, &source, NULL); glCompileShader(s);
  GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) { char log[2048] = ""; glGetShaderInfoLog(s, sizeof log, NULL, log); say("GLRTEST compile log: %s\n", log); }
  return s;
}
static GLuint program(const char *vs, const char *fs, const char *varying) {
  GLuint p = glCreateProgram();
  glAttachShader(p, shader(GL_VERTEX_SHADER, vs)); glAttachShader(p, shader(GL_FRAGMENT_SHADER, fs));
  if (varying) glTransformFeedbackVaryings(p, 1, &varying, GL_INTERLEAVED_ATTRIBS);
  glLinkProgram(p);
  GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) { char log[2048] = ""; glGetProgramInfoLog(p, sizeof log, NULL, log); say("GLRTEST link log: %s\n", log); }
  return ok ? p : 0;
}

static const char *const triangle_vs =
  "#version 300 es\nin vec2 position;\nvoid main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
static const char *const triangle_fs =
  "#version 300 es\nprecision mediump float;\nuniform vec4 webgl_tint;\n"
  "layout(std140) uniform Colors { vec4 base; };\nout vec4 color;\n"
  "void main() { color = base + webgl_tint; }\n";
static const char *const texture_fs =
  "#version 300 es\nprecision mediump float;\nuniform highp sampler2D image;\nout vec4 color;\n"
  "void main() { color = texelFetch(image, ivec2(gl_FragCoord.xy) / 32, 0); }\n";
static const char *const feedback_vs =
  "#version 300 es\nin vec2 position;\nout float doubled;\n"
  "void main() { doubled = position.x * 2.0; gl_Position = vec4(position, 0.0, 1.0); }\n";
static const char *const feedback_fs =
  "#version 300 es\nprecision mediump float;\nout vec4 color;\nvoid main() { color = vec4(1.0); }\n";

int main(void) {
  console = open("/dev/console", O_WRONLY | O_NOCTTY | O_CLOEXEC);
  say("GLRTEST start available=%d\n", glremote_available());
#define LOAD(ret, name, args) name = (ret (*) args)glremote_get_proc_address(#name); \
  if (!name) { say("GLRTEST FAIL missing %s\n", #name); return 1; }
  FUNCS(LOAD)
  EGLDisplay display = eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
  EGLint major = 0, minor = 0;
  if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor)) { say("GLRTEST FAIL egl-initialize 0x%x\n", eglGetError()); return 1; }
  say("GLRTEST egl=%d.%d vendor=\"%s\"\n", major, minor, eglQueryString(display, EGL_VENDOR));
  eglBindAPI(EGL_OPENGL_ES_API);
  static const EGLint config_attribs[] = { EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
                                           EGL_RED_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_NONE };
  EGLConfig config = 0; EGLint count = 0;
  if (!eglChooseConfig(display, config_attribs, &config, 1, &count) || !count) { say("GLRTEST FAIL choose-config 0x%x\n", eglGetError()); return 1; }
  static const EGLint context_attribs[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 0, EGL_NONE };
  double t0 = now_ms();
  EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attribs);
  if (context == EGL_NO_CONTEXT) { say("GLRTEST FAIL create-context 0x%x\n", eglGetError()); return 1; }
  static const EGLint surface_attribs[] = { EGL_WIDTH, 64, EGL_HEIGHT, 64, EGL_NONE };
  EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attribs);
  if (surface == EGL_NO_SURFACE || !eglMakeCurrent(display, surface, surface, context)) { say("GLRTEST FAIL make-current 0x%x\n", eglGetError()); return 1; }
  say("GLRTEST context ms=%.1f\n", now_ms() - t0);
  GLint extensions = 0; glGetIntegerv(GL_NUM_EXTENSIONS, &extensions);
  say("GLRTEST version=\"%s\" renderer=\"%s\" vendor=\"%s\" glsl=\"%s\" extensions=%d\n", glGetString(GL_VERSION),
      glGetString(GL_RENDERER), glGetString(GL_VENDOR), glGetString(GL_SHADING_LANGUAGE_VERSION), extensions);
  say("GLRTEST extension-list %s\n", glGetString(GL_EXTENSIONS));
  check(!strncmp((const char *)glGetString(GL_VERSION), "OpenGL ES 3.0", 13), "es3-version");
  GLint limit = 0; glGetIntegerv(GL_MAX_TEXTURE_SIZE, &limit);
  GLint draw_buffers = 0; glGetIntegerv(GL_MAX_DRAW_BUFFERS, &draw_buffers);
  say("GLRTEST limits tex2d=%d draw_buffers=%d\n", limit, draw_buffers);

  unsigned char pixel[64 * 64 * 4];
  glViewport(0, 0, 64, 64);
  glClearColor(1, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
  glReadPixels(10, 10, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
  check(near(pixel, 255, 0, 0, 255), "clear-readback");

  /* Triangle covering the centre: base (uniform block) + webgl_tint. */
  GLuint p = program(triangle_vs, triangle_fs, NULL);
  check(p != 0, "link-triangle");
  GLuint vao, buffers[3]; glGenVertexArrays(1, &vao); glBindVertexArray(vao); glGenBuffers(3, buffers);
  static const float vertices[] = { -1, -1, 3, -1, -1, 3 };
  glBindBuffer(GL_ARRAY_BUFFER, buffers[0]); glBufferData(GL_ARRAY_BUFFER, sizeof vertices, vertices, GL_STATIC_DRAW);
  GLint position = glGetAttribLocation(p, "position");
  glEnableVertexAttribArray((GLuint)position); glVertexAttribPointer((GLuint)position, 2, GL_FLOAT, GL_FALSE, 0, 0);
  static const float base[4] = { 0, 0.5f, 0, 1 };
  glBindBuffer(GL_UNIFORM_BUFFER, buffers[1]); glBufferData(GL_UNIFORM_BUFFER, sizeof base, base, GL_STATIC_DRAW);
  glUniformBlockBinding(p, glGetUniformBlockIndex(p, "Colors"), 2); glBindBufferBase(GL_UNIFORM_BUFFER, 2, buffers[1]);
  glUseProgram(p);
  GLint tint = glGetUniformLocation(p, "webgl_tint");
  check(tint >= 0, "webgl-prefixed-uniform-location");
  glUniform4f(tint, 0, 0, 0.5f, 0);
  glDrawArrays(GL_TRIANGLES, 0, 3);
  glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
  say("GLRTEST triangle pixel=%d,%d,%d,%d\n", pixel[0], pixel[1], pixel[2], pixel[3]);
  check(near(pixel, 0, 128, 128, 255), "triangle-ubo-uniform");

  /* texelFetch of a 2x2 texture, one texel per 32x32 quadrant. */
  GLuint tp = program(triangle_vs, texture_fs, NULL);
  check(tp != 0, "link-texture");
  GLuint texture; glGenTextures(1, &texture); glBindTexture(GL_TEXTURE_2D, texture);
  static const unsigned char texels[] = { 255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255 };
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 2, 2);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, texels);
  glGetError(); /* TexImage2D on immutable storage is INVALID_OPERATION; respecify mutable instead */
  GLuint texture2; glGenTextures(1, &texture2); glBindTexture(GL_TEXTURE_2D, texture2);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, texels);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glUseProgram(tp); glUniform1i(glGetUniformLocation(tp, "image"), 0);
  glDrawArrays(GL_TRIANGLES, 0, 3);
  glReadPixels(0, 0, 64, 64, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
  check(near(pixel + (8 * 64 + 8) * 4, 255, 0, 0, 255) && near(pixel + (8 * 64 + 40) * 4, 0, 255, 0, 255) &&
        near(pixel + (40 * 64 + 8) * 4, 0, 0, 255, 255) && near(pixel + (40 * 64 + 40) * 4, 255, 255, 255, 255), "texture-texelfetch");

  /* Render into a texture FBO, blit it back. */
  GLuint target, fbo; glGenTextures(1, &target); glBindTexture(GL_TEXTURE_2D, target);
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 64, 64);
  glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, target, 0);
  check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "fbo-complete");
  glClearColor(0, 0, 1, 1); glClear(GL_COLOR_BUFFER_BIT);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo); glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  glBlitFramebuffer(0, 0, 64, 64, 0, 0, 64, 64, GL_COLOR_BUFFER_BIT, GL_NEAREST);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glReadPixels(5, 50, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
  check(near(pixel, 0, 0, 255, 255), "fbo-blit");

  /* Transform feedback read back through MapBufferRange. */
  GLuint fp = program(feedback_vs, feedback_fs, "doubled");
  check(fp != 0, "link-feedback");
  glUseProgram(fp);
  glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buffers[2]);
  glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 3 * sizeof(float), NULL, GL_STREAM_READ);
  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buffers[2]);
  glEnable(GL_RASTERIZER_DISCARD);
  glBeginTransformFeedback(GL_TRIANGLES); glDrawArrays(GL_TRIANGLES, 0, 3); glEndTransformFeedback();
  glDisable(GL_RASTERIZER_DISCARD);
  const float *out = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 3 * sizeof(float), GL_MAP_READ_BIT);
  check(out && out[0] == -2 && out[1] == 6 && out[2] == -2, "transform-feedback");
  if (out) say("GLRTEST feedback %g %g %g\n", out[0], out[1], out[2]);
  glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);

  GLenum error = glGetError();
  check(error == GL_NO_ERROR, "no-gl-error");
  if (error) say("GLRTEST gl-error 0x%x\n", error);

  /* Timing: batched draws (one readback per frame) and plain roundtrips. */
  glUseProgram(p);
  double start = now_ms();
  int frames = 0;
  while (now_ms() - start < 3000) {
    glClear(GL_COLOR_BUFFER_BIT);
    for (int k = 0; k < 100; k++) { glUniform4f(tint, k / 100.0f, 0, 0, 0); glDrawArrays(GL_TRIANGLES, 0, 3); }
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
    frames++;
  }
  double frame_ms = (now_ms() - start) / frames;
  start = now_ms();
  for (int k = 0; k < 200; k++) glGetError();
  double roundtrip_ms = (now_ms() - start) / 200;
  say("GLRTEST timing frames=%d frame_ms=%.3f (100 draws + 1x1 readback) roundtrip_ms=%.3f\n", frames, frame_ms, roundtrip_ms);

  /* Firefox's WebGL present pattern: per frame a draw, then a fresh 320x240 surface
   * checked with GetError/CheckFramebufferStatus, and a full readback. */
  static unsigned char image[320 * 240 * 4];
  GLuint ring_tex[2] = {0}, ring_fbo[2] = {0};
  start = now_ms();
  frames = 0;
  while (now_ms() - start < 3000) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glGetError();
    int slot = frames & 1;
    if (ring_tex[slot]) { glDeleteFramebuffers(1, &ring_fbo[slot]); glDeleteTextures(1, &ring_tex[slot]); }
    glGenTextures(1, &ring_tex[slot]);
    glBindTexture(GL_TEXTURE_2D, ring_tex[slot]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 320, 240, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glGetError();
    glGenFramebuffers(1, &ring_fbo[slot]);
    glBindFramebuffer(GL_FRAMEBUFFER, ring_fbo[slot]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ring_tex[slot], 0);
    glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glReadPixels(0, 0, 320, 240, GL_RGBA, GL_UNSIGNED_BYTE, image);
    frames++;
  }
  say("GLRTEST timing swapchain frames=%d frame_ms=%.3f (draw, 2 GetError, new 320x240 surface, status, readback)\n",
      frames, (now_ms() - start) / frames);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  eglDestroyContext(display, context);
  eglTerminate(display);
  say("GLRTEST DONE %s failures=%d\n", failures ? "FAIL" : "PASS", failures);
  return failures != 0;
}
