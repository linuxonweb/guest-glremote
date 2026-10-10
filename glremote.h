/* Guest OpenGL ES 3.0 + EGL 1.5 forwarded to the host browser's WebGL2.
 *
 * glremote encodes GLES calls into a per-context command stream and sends it
 * over the gpu_hostcall channel (web/linux/dist/guest-gpu-hostcall.js) to the
 * host GPU-owner Worker, which replays it on a WebGL2 context of an
 * OffscreenCanvas (web/linux/dist/guest-gl-remote.js). Calls without a result
 * are batched; queries end the batch and wait for the reply. Object names are
 * allocated in the guest, so glGen* never waits.
 *
 * Every entry point is reachable only through glremote_get_proc_address() (and
 * the EGL table glremote_egl_table), so the library can be linked next to Mesa
 * without symbol clashes; an application picks it at runtime. */
#ifndef GLREMOTE_H
#define GLREMOTE_H

typedef void (*glremote_proc)(void);

/* "glXxx" / "eglXxx" -> function, or NULL. */
glremote_proc glremote_get_proc_address(const char *name);

struct glremote_export { const char *name; glremote_proc fn; };
/* NULL-terminated list of the EGL 1.0-1.5 core entry points. */
extern const struct glremote_export glremote_egl_table[];

/* 1 when the host offers the GL remote (probed once). */
int glremote_available(void);

/* Window-system glue for EGL window surfaces. glremote has no window system
 * of its own: the application installs these before eglInitialize, and
 * without them eglCreateWindowSurface fails with EGL_BAD_NATIVE_WINDOW.
 * `window` is the EGLNativeWindowType value (an X11 Window on X11). A window
 * surface renders into the host context's canvas; eglSwapBuffers reads that
 * back and hands it to present(). */
struct glremote_window_ops {
  /* Current size of the window; 0 when it is gone. */
  int (*size)(unsigned long window, int *width, int *height);
  /* Shows `height` rows, top row first, of `width` little-endian 0xAARRGGBB
   * words at x,y (top-left origin) in the window. Returns 0 on failure. */
  int (*present)(unsigned long window, const unsigned *pixels, int x, int y, int width, int height);
  /* Optional: 1 when the window lost (part of) what it showed since the last
   * call, so the next swap presents the whole frame. */
  int (*exposed)(unsigned long window);
  /* Native visual id with `depth` bits (24 or 32), 0 when there is none. */
  int (*visual)(int depth);
  /* Optional, both or neither: keyed presentation. When the host offers it,
   * the window is filled with a key colour and the host shows the canvas
   * wherever the screen shows that colour, so swaps read nothing back.
   * origin(): the window's position on the root window; 0 when unknown.
   * fill(): paints the whole window with the 0xRRGGBB colour `rgb`; 0 when
   * the window cannot be keyed (gone, or not an opaque 24-bit TrueColor one). */
  int (*origin)(unsigned long window, int *x, int *y);
  int (*fill)(unsigned long window, unsigned rgb);
};
void glremote_set_window_ops(const struct glremote_window_ops *ops);

#endif
