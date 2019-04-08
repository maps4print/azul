use std::{
    mem,
    fmt,
    time::Instant,
    collections::BTreeMap,
    sync::{Arc, Mutex, PoisonError},
};
#[cfg(debug_assertions)]
use azul_css::HotReloadHandler;
use glium::{
    SwapBuffersError,
    glutin::{
        WindowEvent, WindowId as GliumWindowId,
        dpi::{LogicalPosition, LogicalSize}
    },
};
use gleam::gl::{self, Gl, GLuint};
use webrender::{
    PipelineInfo, Renderer,
    api::{
        HitTestResult, HitTestFlags, DevicePixel,
        WorldPoint, LayoutSize, LayoutPoint,
        Epoch, Transaction,
    },
};
#[cfg(feature = "image_loading")]
use app_resources::ImageSource;
#[cfg(feature = "logging")]
use log::LevelFilter;
use azul_css::{Css, ColorU};
use {
    FastHashMap,
    error::ClipboardError,
    window::{
        Window, FakeWindow, ScrollStates,
        WindowCreateError, WindowCreateOptions, RendererType,
    },
    window_state::{WindowSize, DebugState},
    app_resources::TextId,
    dom::{Dom, ScrollTagId},
    app_resources::{
        ImageId, FontSource, FontId, ImageReloadError,
        FontReloadError, CssImageId, RawImage,
    },
    traits::Layout,
    ui_state::UiState,
    ui_description::UiDescription,
    async::{Task, Timer, TimerId, TerminateTimer},
    callbacks::{FocusTarget, UpdateScreen, Redraw, DontRedraw, LayoutInfo},
};
pub use app_resources::AppResources;

type DeviceIntSize = ::euclid::TypedSize2D<i32, DevicePixel>;

// Default clear color is white, to signify that there is rendering going on
// (otherwise, "transparent") backgrounds would be painted black.
const COLOR_WHITE: ColorU = ColorU { r: 255, g: 255, b: 255, a: 0 };

/// Graphical application that maintains some kind of application state
pub struct App<T> {
    /// The graphical windows, indexed by their system ID / handle
    windows: BTreeMap<GliumWindowId, Window<T>>,
    /// The global application state
    pub app_state: AppState<T>,
    /// Application configuration, whether to enable logging, etc.
    pub config: AppConfig,
    /// The `Layout::layout()` callback, stored as a function pointer,
    /// There are multiple reasons for doing this (instead of requiring `T: Layout` everywhere):
    ///
    /// - It seperates the `Dom<T>` from the `Layout` trait, making it possible to split the UI solving and styling into reusable crates
    /// - It's less typing work (prevents having to type `<T: Layout>` everywhere)
    /// - It's potentially more efficient to compile (less type-checking required)
    /// - It's a preparation for the C ABI, in which traits don't exist (for language bindings).
    ///   In the C ABI "traits" are simply structs with function pointers (and void* instead of T)
    layout_callback: fn(&T, layout_info: LayoutInfo<T>) -> Dom<T>,
}

/// Configuration for optional features, such as whether to enable logging or panic hooks
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "logging"), derive(Copy))]
pub struct AppConfig {
    /// If enabled, logs error and info messages.
    ///
    /// Default is `Some(LevelFilter::Error)` to log all errors by default
    #[cfg(feature = "logging")]
    pub enable_logging: Option<LevelFilter>,
    /// Path to the output log if the logger is enabled
    #[cfg(feature = "logging")]
    pub log_file_path: Option<String>,
    /// If the app crashes / panics, a window with a message box pops up.
    /// Setting this to `false` disables the popup box.
    #[cfg(feature = "logging")]
    pub enable_visual_panic_hook: bool,
    /// If this is set to `true` (the default), a backtrace + error information
    /// gets logged to stdout and the logging file (only if logging is enabled).
    #[cfg(feature = "logging")]
    pub enable_logging_on_panic: bool,
    /// (STUB) Whether keyboard navigation should be enabled (default: true).
    /// Currently not implemented.
    pub enable_tab_navigation: bool,
    /// Whether to force a hardware or software renderer
    pub renderer_type: RendererType,
    /// Debug state for all windows
    pub debug_state: DebugState,
    /// Background color for all windows
    pub background_color: ColorU,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            #[cfg(feature = "logging")]
            enable_logging: Some(LevelFilter::Error),
            #[cfg(feature = "logging")]
            log_file_path: None,
            #[cfg(feature = "logging")]
            enable_visual_panic_hook: true,
            #[cfg(feature = "logging")]
            enable_logging_on_panic: true,
            enable_tab_navigation: true,
            renderer_type: RendererType::default(),
            debug_state: DebugState::default(),
            background_color: COLOR_WHITE,
        }
    }
}

/// Wrapper for your application data, stores the data, windows and resources, as
/// well as running timers and asynchronous tasks.
///
/// In order to be layout-able, your data model needs to satisfy the `Layout` trait,
/// which maps the state of your application to a DOM (how the application data should be laid out)
pub struct AppState<T> {
    /// Your data (the global struct which all callbacks will have access to)
    pub data: Arc<Mutex<T>>,
    /// This field represents the state of the windows, public to the user. You can
    /// mess around with the state as you like, however, the actual window won't update
    /// until the next frame. This is done to "decouple" the frameworks internal
    /// state updating logic from the user code (and to make the API future-proof
    /// in case extra functions are introduced).
    ///
    /// Another reason this is needed is to (later) introduce testing for the window
    /// state - if the API would directly modify the window itself, these changes
    /// wouldn't be recorded anywhere, so there wouldn't be a way to unit-test certain APIs.
    ///
    /// The state of these `FakeWindow`s gets deleted and recreated on each frame, especially
    /// the app's style. This should force a user to design his code in a functional way,
    /// without relying on state-based conditions. Example:
    ///
    /// ```no_run,ignore
    /// let window_state = &mut app_state.windows[event.window];
    /// // Update the title
    /// window_state.state.title = "Hello";
    /// ```
    pub windows: BTreeMap<GliumWindowId, FakeWindow<T>>,
    /// Fonts, images and cached text that is currently loaded inside the app (window-independent).
    ///
    /// Accessing this field is often required to load new fonts or images, so instead of
    /// requiring the `FontHashMap`, a lot of functions just require the whole `AppResources` field.
    pub resources: AppResources,
    /// Currently running timers (polling functions, run on the main thread)
    pub(crate) timers: FastHashMap<TimerId, Timer<T>>,
    /// Currently running tasks (asynchronous functions running each on a different thread)
    pub(crate) tasks: Vec<Task<T>>,
}

/// Same as the [AppState](./struct.AppState.html) but without the
/// `self.data` field - used for default callbacks, so that callbacks can
/// load and unload fonts or images + access the system clipboard
///
/// Default callbacks don't have access to the `AppState.data` field,
/// since they use a `StackCheckedPointer` instead.
pub struct AppStateNoData<'a, T> {
    /// See [`AppState.windows`](./struct.AppState.html#structfield.windows)
    pub windows: &'a BTreeMap<GliumWindowId, FakeWindow<T>>,
    /// See [`AppState.resources`](./struct.AppState.html#structfield.resources)
    pub resources : &'a mut AppResources,
    /// Currently running timers (polling functions, run on the main thread)
    pub(crate) timers: FastHashMap<TimerId, Timer<T>>,
    /// Currently running tasks (asynchronous functions running each on a different thread)
    pub(crate) tasks: Vec<Task<T>>,
}

/// Error returned by the `.run()` function
///
/// If the `.run()` function would panic, that would need `T` to
/// implement `Debug`, which is not necessary if we just return an error.
pub enum RuntimeError<T> {
    // Could not swap the display (drawing error)
    GlSwapError(SwapBuffersError),
    ArcUnlockError,
    MutexPoisonError(PoisonError<T>),
    MutexLockError,
    WindowIndexError,
}

pub(crate) struct FrameEventInfo {
    pub(crate) should_redraw_window: bool,
    pub(crate) should_hittest: bool,
    pub(crate) cur_cursor_pos: LogicalPosition,
    pub(crate) new_window_size: Option<LogicalSize>,
    pub(crate) new_dpi_factor: Option<f64>,
    pub(crate) is_resize_event: bool,
}

impl Default for FrameEventInfo {
    fn default() -> Self {
        Self {
            should_redraw_window: false,
            should_hittest: false,
            cur_cursor_pos: LogicalPosition::new(0.0, 0.0),
            new_window_size: None,
            new_dpi_factor: None,
            is_resize_event: false,
        }
    }
}

impl<T> From<PoisonError<T>> for RuntimeError<T> {
    fn from(e: PoisonError<T>) -> Self {
        RuntimeError::MutexPoisonError(e)
    }
}

impl<T> From<SwapBuffersError> for RuntimeError<T> {
    fn from(e: SwapBuffersError) -> Self {
        RuntimeError::GlSwapError(e)
    }
}

impl<T> fmt::Debug for RuntimeError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::RuntimeError::*;
        match self {
            GlSwapError(e) => write!(f, "Failed to swap GL display: {}", e),
            ArcUnlockError => write!(f, "Failed to unlock arc on application shutdown"),
            MutexPoisonError(e) => write!(f, "Mutex poisoned (thread panicked unexpectedly): {}", e),
            MutexLockError => write!(f, "Failed to lock application state mutex"),
            WindowIndexError => write!(f, "Invalid window index"),
        }
    }
}

impl<T> fmt::Display for RuntimeError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self))
    }
}

impl<'a, T: 'a> AppStateNoData<'a, T> {
    impl_deamon_api!();
}

impl<T: Layout> App<T> {

    #[allow(unused_variables)]
    /// Create a new, empty application. This does not open any windows.
    pub fn new(initial_data: T, config: AppConfig) -> Result<Self, WindowCreateError> {

        #[cfg(feature = "logging")] {
            if let Some(log_level) = config.enable_logging {
                ::logging::set_up_logging(config.log_file_path.as_ref().map(|s| s.as_str()), log_level);

                if config.enable_logging_on_panic {
                    ::logging::set_up_panic_hooks();
                }

                if config.enable_visual_panic_hook {
                    use std::sync::atomic::Ordering;
                    ::logging::SHOULD_ENABLE_PANIC_HOOK.store(true, Ordering::SeqCst);
                }
            }
        }

        let mut app_state = AppState::new(initial_data, &config)?;

        if let Some(r) = &mut app_state.resources.fake_display.renderer {
            set_webrender_debug_flags(r, &DebugState::default(), &config.debug_state);
        }

        Ok(Self {
            windows: BTreeMap::new(),
            app_state,
            config,
            layout_callback: T::layout,
        })
    }
}

impl<T> App<T> {

    /// Creates a new window
    pub fn create_window(&mut self, options: WindowCreateOptions<T>, css: Css)
    -> Result<Window<T>, WindowCreateError>
    {
        Window::new(
            &mut self.app_state.resources.fake_display.render_api,
            &mut self.app_state.resources.fake_display.hidden_display.gl_window().context(),
            &mut self.app_state.resources.fake_display.hidden_events_loop,
            options,
            css,
            self.config.background_color,
        )
    }

    #[cfg(debug_assertions)]
    pub fn create_hot_reload_window(&mut self, options: WindowCreateOptions<T>, css_loader: Box<dyn HotReloadHandler>)
    -> Result<Window<T>, WindowCreateError>
    {
        Window::new_hot_reload(
            &mut self.app_state.resources.fake_display.render_api,
            &mut self.app_state.resources.fake_display.hidden_display.gl_window().context(),
            &mut self.app_state.resources.fake_display.hidden_events_loop,
            options,
            css_loader,
            self.config.background_color,
        )
    }

    /// Spawn a new window on the screen. Note that this should only be used to
    /// create extra windows, the default window will be the window submitted to
    /// the `.run` method.
    pub fn add_window(&mut self, window: Window<T>) {
        use callbacks::DefaultCallbackSystem;

        let window_id = window.id;
        let fake_window = FakeWindow {
            state: window.state.clone(),
            default_callbacks: DefaultCallbackSystem::new(),
            read_only_window: window.display.clone(),
        };

        self.app_state.windows.insert(window_id, fake_window);
        self.windows.insert(window_id, window);
    }

    /// Start the rendering loop for the currently open windows
    /// This is the "main app loop", "main game loop" or whatever you want to call it.
    /// Usually this is the last function you call in your `main()` function, since exiting
    /// it means that the user has closed all windows and wants to close the app.
    ///
    /// When all windows are closed, this function returns the internal data again.
    /// This is useful for ex. CLI application that run procedurally, but then want to
    /// open a window temporarily, to ask for user input in a "nicer" way than a pure
    /// CLI-way.
    ///
    /// This way you can do this:
    ///
    /// ```no_run,ignore
    /// let app = App::new(MyData { username: None, password: None });
    /// app.create_window(WindowCreateOptions::default(), azul_native_style::native());
    ///
    /// // pop open a window that asks the user for his username and password...
    /// let MyData { username, password } = app.run();
    ///
    /// // continue the rest of the program here...
    /// println!("username: {:?}, password: {:?}", username, password);
    /// ```
    pub fn run(mut self, window: Window<T>) -> Result<T, RuntimeError<T>> {

        // Apps need to have at least one window open
        self.add_window(window);
        self.run_inner()?;

        // NOTE: This is necessary because otherwise, the Arc::try_unwrap would fail,
        // since one Arc is still owned by the app_state.tasks structure
        //
        // See https://github.com/maps4print/azul/issues/24#issuecomment-429737273
        mem::drop(self.app_state.tasks);

        let unique_arc = Arc::try_unwrap(self.app_state.data).map_err(|_| RuntimeError::ArcUnlockError)?;
        unique_arc.into_inner().map_err(|e| e.into())
    }

    fn run_inner(&mut self) -> Result<(), RuntimeError<T>> {

        use std::{thread, time::Duration};
        use glium::glutin::Event;
        use self::RuntimeError::*;

        let mut ui_state_cache = {
            let app_state = &mut self.app_state;
            let mut ui_state_map = BTreeMap::new();
            for window_id in self.windows.keys() {
              ui_state_map.insert(*window_id, UiState::from_app_state(app_state, window_id, self.layout_callback)?);
            }
            ui_state_map
        };
        let mut ui_description_cache = self.windows.keys().map(|window_id| (*window_id, UiDescription::default())).collect();
        let mut force_redraw_cache = self.windows.keys().map(|window_id| (*window_id, 2)).collect();
        let mut awakened_task = self.windows.keys().map(|window_id| (*window_id, false)).collect();

        #[cfg(debug_assertions)]
        let mut last_style_reload = Instant::now();
        #[cfg(debug_assertions)]
        let mut should_print_css_error = true;

        while !self.windows.is_empty() {

            let time_start = Instant::now();

            let mut closed_windows = Vec::<GliumWindowId>::new();
            let mut frame_was_resize = false;
            let mut events = Vec::new();

            self.app_state.resources.fake_display.hidden_events_loop.poll_events(|e| events.push(e));

            // let current_desktop_events = get_desktop_events(window, &events);

            for (current_window_id, mut window) in self.windows.iter_mut() {

                // Only process the events belong to this window ID...
                let window_events = events.iter().cloned().filter_map(|e| match e {
                    Event::WindowEvent { window_id, event } => {
                        if *current_window_id == window_id { Some(event) } else { None }
                    },
                    _ => None
                }).collect::<Vec<WindowEvent>>();

                let (event_was_resize, window_was_closed) =
                    render_single_window_content(
                        self.layout_callback,
                        &self.config,
                        &window_events,
                        &current_window_id,
                        &mut window,
                        &mut self.app_state,
                        &mut ui_state_cache,
                        &mut ui_description_cache,
                        &mut force_redraw_cache,
                        &mut awakened_task,
                    )?;

                if event_was_resize {
                    frame_was_resize = true;
                }

                if window_was_closed {
                    closed_windows.push(*current_window_id);

                    // TODO: Currently there is no way to return from the main event loop
                    // i.e. the windows aren't actually getting closed
                    // This is a hack, so that windows currently close properly
                    return Ok(());
                }
            }

            #[cfg(debug_assertions)] {
                hot_reload_css(
                    &mut self.windows,
                    &mut last_style_reload,
                    &mut should_print_css_error,
                    &mut awakened_task
                )?;
            }

            // Close windows if necessary
            closed_windows.into_iter().for_each(|closed_window_id| {
                ui_state_cache.remove(&closed_window_id);
                ui_description_cache.remove(&closed_window_id);
                force_redraw_cache.remove(&closed_window_id);
                self.windows.remove(&closed_window_id);
            });

            let should_redraw_timers = self.app_state.run_all_timers();
            let should_redraw_tasks = self.app_state.clean_up_finished_tasks();
            let should_redraw_timers_or_tasks = [should_redraw_timers, should_redraw_tasks].into_iter().any(|e| *e == Redraw);

            if should_redraw_timers_or_tasks {
                // self.windows.iter().for_each(|(_, window)| {
                //     app_state.resources.fake_display.events_loop.create_proxy().wakeup().unwrap_or(())
                // });
                awakened_task = self.windows.keys().map(|window_id| (*window_id, true)).collect();
                for window_id in self.windows.keys() {
                    *force_redraw_cache.get_mut(window_id).ok_or(WindowIndexError)? = 2;
                }
            }

            // Automatically remove unused fonts and images from webrender
            // Tell the font + image GC to start a new frame
            self.app_state.resources.garbage_collect_fonts_and_images();

            if !frame_was_resize {
                // Wait until 16ms have passed, but not during a resize event
                let diff = time_start.elapsed();
                const FRAME_TIME: Duration = Duration::from_millis(16);
                if diff < FRAME_TIME {
                    thread::sleep(FRAME_TIME - diff);
                }
            }
        }

        Ok(())
    }

    /// See `AppState::add_task`.
    pub fn add_task(&mut self, task: Task<T>) {
        self.app_state.add_task(task);
    }

    /// Toggles debugging flags in webrender, updates `self.config.debug_state`
    pub fn toggle_debug_flags(&mut self, new_state: DebugState) {
        if let Some(r) = &mut self.app_state.resources.fake_display.renderer {
            set_webrender_debug_flags(r, &self.config.debug_state, &new_state);
        }
        self.config.debug_state = new_state;
    }
}

image_api!(App::app_state);
font_api!(App::app_state);
text_api!(App::app_state);
clipboard_api!(App::app_state);
timer_api!(App::app_state);

impl<T> AppState<T> {

    /// Creates a new `AppState`
    fn new(initial_data: T, config: &AppConfig) -> Result<Self, WindowCreateError> {
        Ok(Self {
            data: Arc::new(Mutex::new(initial_data)),
            windows: BTreeMap::new(),
            resources: AppResources::new(config)?,
            timers: FastHashMap::default(),
            tasks: Vec::new(),
        })
    }

    impl_deamon_api!();

    /// Run all currently registered timers
    #[must_use]
    fn run_all_timers(&mut self) -> UpdateScreen {
        let mut should_update_screen = DontRedraw;
        let mut lock = self.data.lock().unwrap();
        let mut timers_to_terminate = Vec::new();

        for (key, timer) in self.timers.iter_mut() {
            let (should_update, should_terminate) = timer.invoke_callback_with_data(&mut lock, &mut self.resources);

            if should_update == Redraw &&
               should_update_screen == DontRedraw {
                should_update_screen = Redraw;
            }

            if should_terminate == TerminateTimer::Terminate {
                timers_to_terminate.push(key.clone());
            }
        }

        for key in timers_to_terminate {
            self.timers.remove(&key);
        }

        should_update_screen
    }

    /// Remove all tasks that have finished executing
    #[must_use] fn clean_up_finished_tasks(&mut self) -> UpdateScreen {
        let old_count = self.tasks.len();
        let mut timers_to_add = Vec::new();
        self.tasks.retain(|task| {
            if task.is_finished() {
                if let Some(timer) = task.after_completion_timer {
                    timers_to_add.push((TimerId::new(), timer));
                }
                false
            } else {
                true
            }
        });

        let timers_is_empty = timers_to_add.is_empty();
        let new_count = self.tasks.len();

        // Start all the timers that should run after the completion of the task
        for (timer_id, timer) in timers_to_add {
            self.add_timer(timer_id, timer);
        }

        if old_count == new_count && timers_is_empty {
            DontRedraw
        } else {
            Redraw
        }
    }
}

image_api!(AppState::resources);
font_api!(AppState::resources);
text_api!(AppState::resources);
clipboard_api!(AppState::resources);

/// Render the contents of one single window.
/// Returns (if the event was a resize event, if the window was closed)
fn render_single_window_content<T>(
    layout_callback: fn(&T, LayoutInfo<T>) -> Dom<T>,
    config: &AppConfig,
    events: &[WindowEvent],
    window_id: &GliumWindowId,
    window: &mut Window<T>,
    app_state: &mut AppState<T>,
    ui_state_cache: &mut BTreeMap<GliumWindowId, UiState<T>>,
    ui_description_cache: &mut BTreeMap<GliumWindowId, UiDescription<T>>,
    force_redraw_cache: &mut BTreeMap<GliumWindowId, usize>,
    awakened_task: &mut BTreeMap<GliumWindowId, bool>,
) -> Result<(bool, bool), RuntimeError<T>> {

    use self::RuntimeError::*;

    if events.is_empty() && force_redraw_cache[window_id] == 0 {
        // Event was not a resize event, window should **not** close
        return Ok((false, false));
    }

    let (mut frame_event_info, window_should_close) = window.state.update_window_state(&events);

    if window_should_close {
        // Event was not a resize event, window should close
        return Ok((false, true));
    }

    let mut hit_test_results = None;

    if frame_event_info.should_hittest {

        hit_test_results = do_hit_test(&window, &app_state.resources);

        for event in events.iter() {

            let callback_result = call_callbacks(
                hit_test_results.as_ref(),
                event,
                window,
                &window_id,
                &ui_state_cache[&window_id],
                app_state
            )?;

            if callback_result.should_update_screen == Redraw {
                frame_event_info.should_redraw_window = true;
            }

            // Note: Don't set `pending_focus_target` directly here, because otherwise
            // callbacks that return `Some()` would get immediately overwritten again
            // by callbacks that return `None`.
            if let Some(overwrites_focus) = callback_result.callbacks_overwrites_focus {
                window.state.internal.pending_focus_target = Some(overwrites_focus);
            }
        }
    }

    // Scroll for the scrolled amount for each node that registered a scroll state.
    let should_scroll_render = update_scroll_state(window, hit_test_results, &mut app_state.resources);

    if frame_event_info.is_resize_event {
        // This is a hack because during a resize event, winit eats the "awakened"
        // event. So what we do is that we call the layout-and-render again, to
        // trigger a second "awakened" event. So when the window is resized, the
        // layout function is called twice (the first event will be eaten by winit)
        //
        // This is a reported bug and should be fixed somewhere in July
        *force_redraw_cache.get_mut(window_id).ok_or(WindowIndexError)? = 2;
    }

    // See: https://docs.rs/glutin/0.19.0/glutin/struct.CombinedContext.html#method.resize
    //
    // Some platforms (macOS, Wayland) require being manually updated when their window
    // or surface is resized.
    #[cfg(not(target_os = "windows"))] {
        if frame_event_info.is_resize_event {
            // Resize gl window
            let gl_window = window.display.gl_window();
            let size = gl_window.get_inner_size().unwrap().to_physical(gl_window.get_hidpi_factor());
            gl_window.resize(size);
        }
    }

    // Update the window state that we got from the frame event (updates window dimensions and DPI)
    // Sets frame_event_info.needs redraw if the event was a
    window.update_from_external_window_state(&mut frame_event_info, &app_state.resources.fake_display.hidden_events_loop);
    // Update the window state every frame that was set by the user
    window.update_from_user_window_state(app_state.windows[&window_id].state.clone());
    // Reset the scroll amount to 0 (for the next frame)
    window.clear_scroll_state();

    // If there is already a layout construction in progress, prevent
    // re-rendering on layout, otherwise this leads to jankiness during scrolling
    let should_relayout = frame_event_info.should_redraw_window || awakened_task[window_id] || force_redraw_cache[window_id] > 0;
    let should_rerender = should_scroll_render || frame_event_info.is_resize_event;

    if should_relayout || should_rerender {

        // Call the Layout::layout() fn, get the DOM
        *ui_state_cache.get_mut(window_id).ok_or(WindowIndexError)? =
            UiState::from_app_state(app_state, window_id, layout_callback)?;

        // Style the DOM (is_mouse_down is necessary for styling :hover, :active + :focus nodes)
        let is_mouse_down = window.state.internal.mouse_state.mouse_down();

        *ui_description_cache.get_mut(window_id).ok_or(WindowIndexError)? =
            UiDescription::match_css_to_dom(
                ui_state_cache.get_mut(window_id).ok_or(WindowIndexError)?,
                &window.css,
                &mut window.state.internal.focused_node,
                &mut window.state.internal.pending_focus_target,
                &window.state.internal.hovered_nodes,
                is_mouse_down,
            );

        // Render the window (webrender will send an Awakened event when the frame is done)
        let mut fake_window = app_state.windows.get_mut(window_id).ok_or(WindowIndexError)?;
        update_display_list(
            &mut app_state.data,
            &ui_description_cache[window_id],
            &ui_state_cache[window_id],
            &mut *window,
            &mut fake_window,
            &mut app_state.resources,
        );
        *awakened_task.get_mut(window_id).ok_or(WindowIndexError)? = false;

        if let Some(i) = force_redraw_cache.get_mut(window_id) {
            if *i > 0 { *i -= 1 };
            if *i == 1 {
                clean_up_unused_opengl_textures(app_state.resources.fake_display.renderer.as_mut().unwrap().flush_pipeline_info());
            }
        }
    }

    // TODO: Render all windows again, not just this one!
    // if should_relayout || should_rerender {
        render_inner(window, &mut app_state.resources, Transaction::new(), config.background_color);
    // }

    Ok((frame_event_info.is_resize_event, false))
}

/// Returns if there was an error with the CSS reloading, necessary so that the error message is only printed once
#[cfg(debug_assertions)]
fn hot_reload_css<T>(
    windows: &mut BTreeMap<GliumWindowId, Window<T>>,
    last_style_reload: &mut Instant,
    should_print_error: &mut bool,
    awakened_tasks: &mut BTreeMap<GliumWindowId, bool>)
-> Result<(), RuntimeError<T>>
{
    use self::RuntimeError::*;
    for (window_id, window) in windows.iter_mut() {
        // Hot-reload a style if necessary
        let hot_reloader = match window.css_loader.as_mut() {
            None => continue,
            Some(s) => s,
        };

        let should_reload = Instant::now() - *last_style_reload > hot_reloader.get_reload_interval();

        if !should_reload {
            continue;
        }

        match hot_reloader.reload_style() {
            Ok(mut new_css) => {
                new_css.sort_by_specificity();
                window.css = new_css;
                if !(*should_print_error) {
                    println!("--- OK: CSS parsed without errors, continuing hot-reload.");
                }
                *last_style_reload = Instant::now();
                // window.events_loop.create_proxy().wakeup().unwrap_or(());
                *awakened_tasks.get_mut(window_id).ok_or(WindowIndexError)? = true;

                *should_print_error = true;
            },
            Err(why) => {
                if *should_print_error {
                    println!("{}", why);
                }
                *should_print_error = false;
            },
        };
    }

    Ok(())
}

/// Returns the currently hit-tested results, in back-to-front order
fn do_hit_test<T>(window: &Window<T>, app_resources: &AppResources) -> Option<HitTestResult> {

    let cursor_location = window.state.internal.mouse_state.cursor_pos
        .map(|pos| WorldPoint::new(pos.x as f32, pos.y as f32))?;

    let mut hit_test_results = app_resources.fake_display.render_api.hit_test(
        window.internal.document_id,
        Some(window.internal.pipeline_id),
        cursor_location,
        HitTestFlags::FIND_ALL
    );

    // Execute callbacks back-to-front, not front-to-back
    hit_test_results.items.reverse();

    Some(hit_test_results)
}

/// Struct returned from the `call_callbacks()` function -
/// returns important information from the callbacks
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CallCallbackReturn {
    /// Whether one or more callbacks say to redraw the screen or not
    pub should_update_screen: UpdateScreen,
    /// Whether one or more callbacks have messed with the current
    /// focused element i.e. via `.clear_focus()` or similar.
    pub callbacks_overwrites_focus: Option<FocusTarget>,
}

/// Returns an bool whether the window should be redrawn or not (true - redraw the screen, false: don't redraw).
fn call_callbacks<T>(
    hit_test_results: Option<&HitTestResult>,
    event: &WindowEvent,
    window: &mut Window<T>,
    window_id: &GliumWindowId,
    ui_state: &UiState<T>,
    app_state: &mut AppState<T>)
-> Result<CallCallbackReturn, RuntimeError<T>>
{
    use {
        callbacks::CallbackInfo,
        window_state::{KeyboardState, MouseState},
        self::RuntimeError::*,
    };

    let mut should_update_screen = DontRedraw;

    let hit_test_items = hit_test_results.map(|h| h.items.clone()).unwrap_or_default();

    let callbacks_filter_list = window.state.determine_callbacks(&hit_test_items, event, ui_state);

    // TODO: this should be refactored - currently very stateful and error-prone!
    app_state.windows.get_mut(window_id).ok_or(WindowIndexError)?
        .set_keyboard_state(&window.state.internal.keyboard_state);
    app_state.windows.get_mut(window_id).ok_or(WindowIndexError)?
        .set_mouse_state(&window.state.internal.mouse_state);

    let mut callbacks_overwrites_focus = None;

    let mut default_timers = FastHashMap::default();
    let mut default_tasks = Vec::new();

    // Run all default callbacks - **before** the user-defined callbacks are run!
    {
        let mut lock = app_state.data.lock().map_err(|_| RuntimeError::MutexLockError)?;

        for (node_id, callback_results) in callbacks_filter_list.nodes_with_callbacks.iter() {
            let hit_item = &callback_results.hit_test_item;
            for default_callback_id in callback_results.default_callbacks.values() {

                let mut callback_info = CallbackInfo {
                    focus: None,
                    window_id,
                    hit_dom_node: *node_id,
                    ui_state,
                    hit_test_items: &hit_test_items,
                    cursor_relative_to_item: hit_item.as_ref().map(|hi| (hi.point_relative_to_item.x, hi.point_relative_to_item.y)),
                    cursor_in_viewport: hit_item.as_ref().map(|hi| (hi.point_in_viewport.x, hi.point_in_viewport.y)),
                };

                let mut app_state_no_data = AppStateNoData {
                    windows: &app_state.windows,
                    resources: &mut app_state.resources,
                    timers: FastHashMap::default(),
                    tasks: Vec::new(),
                };

                if app_state.windows[window_id].default_callbacks.run_callback(
                    &mut *lock,
                    default_callback_id,
                    &mut app_state_no_data,
                    &mut callback_info
                ) == Redraw {
                    should_update_screen = Redraw;
                }

                default_timers.extend(app_state_no_data.timers.into_iter());
                default_tasks.extend(app_state_no_data.tasks.into_iter());

                // Overwrite the focus from the callback info
                if let Some(new_focus) = callback_info.focus {
                    callbacks_overwrites_focus = Some(new_focus);
                }
            }
        }
    }

    // If the default callbacks have started timers or tasks, add them to the main app state
    for (timer_id, timer) in default_timers {
        app_state.add_timer(timer_id, timer);
    }

    for task in default_tasks {
        app_state.add_task(task);
    }

    for (node_id, callback_results) in callbacks_filter_list.nodes_with_callbacks.iter() {
        let hit_item = &callback_results.hit_test_item;
        for callback in callback_results.normal_callbacks.values() {

            let mut callback_info = CallbackInfo {
                focus: None,
                window_id,
                hit_dom_node: *node_id,
                ui_state: &ui_state,
                hit_test_items: &hit_test_items,
                cursor_relative_to_item: hit_item.as_ref().map(|hi| (hi.point_relative_to_item.x, hi.point_relative_to_item.y)),
                cursor_in_viewport: hit_item.as_ref().map(|hi| (hi.point_in_viewport.x, hi.point_in_viewport.y)),
            };

            if (callback.0)(app_state, &mut callback_info) == Redraw {
                should_update_screen = Redraw;
            }

            if let Some(new_focus) = callback_info.focus {
                callbacks_overwrites_focus = Some(new_focus);
            }
        }
    }

    if callbacks_filter_list.needs_redraw_anyways {
        should_update_screen = Redraw;
    }

    app_state.windows.get_mut(window_id).ok_or(WindowIndexError)?
        .set_keyboard_state(&KeyboardState::default());
    app_state.windows.get_mut(window_id).ok_or(WindowIndexError)?
        .set_mouse_state(&MouseState::default());

    Ok(CallCallbackReturn {
        should_update_screen,
        callbacks_overwrites_focus,
    })
}

/// Build the display list and send it to webrender
fn update_display_list<T>(
    app_data: &mut Arc<Mutex<T>>,
    ui_description: &UiDescription<T>,
    ui_state: &UiState<T>,
    window: &mut Window<T>,
    fake_window: &mut FakeWindow<T>,
    app_resources: &mut AppResources,
) {
    use display_list::DisplayList;

    let display_list = DisplayList::new_from_ui_description(ui_description, ui_state);

    // NOTE: layout_result contains all words, text information, etc.
    // - very important for selection!
    let (builder, scrolled_nodes, _layout_result) = display_list.into_display_list_builder(
        app_data,
        window,
        fake_window,
        app_resources,
    );

    // NOTE: Display list has to be rebuilt every frame, otherwise, the epochs get out of sync
    let display_list_builder = builder.finalize().2;
    window.internal.last_scrolled_nodes = scrolled_nodes;

    let (logical_size, _) = convert_window_size(&window.state.size);

    let mut txn = Transaction::new();
    txn.set_display_list(
        window.internal.epoch,
        None,
        logical_size.clone(),
        (window.internal.pipeline_id, logical_size, display_list_builder),
        true,
    );

    app_resources.fake_display.render_api.send_transaction(window.internal.document_id, txn);
}

/// Scroll all nodes in the ScrollStates to their correct position and insert
/// the positions into the transaction
///
/// NOTE: scroll_states has to be mutable, since every key has a "visited" field, to
/// indicate whether it was used during the current frame or not.
fn scroll_all_nodes(scroll_states: &mut ScrollStates, txn: &mut Transaction) {
    use webrender::api::ScrollClamping;
    for (key, value) in scroll_states.0.iter_mut() {
        let (x, y) = value.get();
        txn.scroll_node_with_id(LayoutPoint::new(x, y), *key, ScrollClamping::ToContentBounds);
    }
}

/// Returns the (logical_size, physical_size) as LayoutSizes, which can then be passed to webrender
fn convert_window_size(size: &WindowSize) -> (LayoutSize, DeviceIntSize) {
    let logical_size = LayoutSize::new(
        (size.dimensions.width * size.winit_hidpi_factor) as f32,
        (size.dimensions.height * size.winit_hidpi_factor)  as f32
    );
    let physical_size = size.dimensions.to_physical(size.winit_hidpi_factor);
    let physical_size = DeviceIntSize::new(physical_size.width as i32, physical_size.height as i32);
    (logical_size, physical_size)
}

/// Special rendering function that skips building a layout and only does
/// hit-testing and rendering - called on pure scroll events, since it's
/// significantly less CPU-intensive to just render the last display list instead of
/// re-layouting on every single scroll event.
#[must_use]
fn update_scroll_state<T>(
    window: &mut Window<T>,
    hit_test_results: Option<HitTestResult>,
    app_resources: &mut AppResources,
) -> bool {

    const SCROLL_THRESHOLD: f64 = 0.5; // px

    let hit_test_results = match hit_test_results {
        Some(s) => s,
        None => match do_hit_test(&window, app_resources) {
            Some(s) => s,
            None => return false,
        }
    };

    let scroll_x = window.state.internal.mouse_state.scroll_x;
    let scroll_y = window.state.internal.mouse_state.scroll_y;

    if scroll_x.abs() < SCROLL_THRESHOLD && scroll_y.abs() < SCROLL_THRESHOLD {
        return false;
    }

    let mut should_scroll_render = false;

    let scrolled_nodes = &window.internal.last_scrolled_nodes;
    let scroll_states = &mut window.scroll_states;

    for scroll_node in hit_test_results.items.iter()
        .filter_map(|item| scrolled_nodes.tags_to_node_ids.get(&ScrollTagId(item.tag.0)))
        .filter_map(|node_id| scrolled_nodes.overflowing_nodes.get(&node_id)) {

        // The external scroll ID is constructed from the DOM hash
        let scroll_id = scroll_node.parent_external_scroll_id;

        if scroll_states.0.contains_key(&scroll_id) {
            // TODO: make scroll speed configurable (system setting?)
            scroll_states.scroll_node(&scroll_id, scroll_x as f32, scroll_y as f32);
            should_scroll_render = true;
        }
    }

    should_scroll_render
}

fn clean_up_unused_opengl_textures(pipeline_info: PipelineInfo) {

    use compositor::ACTIVE_GL_TEXTURES;

    // TODO: currently active epochs can be empty, why?
    //
    // I mean, while the renderer is rendering, there can never be "no epochs" active,
    // at least one epoch must always be active.
    if pipeline_info.epochs.is_empty() {
        return;
    }

    // TODO: pipeline_info.epochs does not contain all active epochs,
    // at best it contains the lowest in-use epoch. I.e. if `Epoch(43)`
    // is listed, you can remove all textures from Epochs **lower than 43**
    // BUT NOT EPOCHS HIGHER THAN 43.
    //
    // This means that "all active epochs" (in the documentation) is misleading
    // since it doesn't actually list all active epochs, otherwise it'd list Epoch(43),
    // Epoch(44), Epoch(45), which are currently active.
    let oldest_to_remove_epoch = pipeline_info.epochs.values().min().unwrap();

    let mut active_textures_lock = ACTIVE_GL_TEXTURES.lock().unwrap();

    // Retain all OpenGL textures from epochs higher than the lowest epoch
    //
    // TODO: Handle overflow of Epochs correctly (low priority)
    active_textures_lock.retain(|key, _| key > oldest_to_remove_epoch);
}

// We don't want the epoch to increase to u32::MAX, since
// u32::MAX represents an invalid epoch, which could confuse webrender
fn increase_epoch(old: Epoch) -> Epoch {
    use std::u32;
    const MAX_ID: u32 = u32::MAX - 1;
    match old.0 {
        MAX_ID => Epoch(0),
        other => Epoch(other + 1),
    }
}

// Function wrapper that is invoked on scrolling and normal rendering - only renders the
// window contents and updates the screen, assumes that all transactions via the RenderApi
// have been committed before this function is called.
//
// WebRender doesn't reset the active shader back to what it was, but rather sets it
// to zero, which glium doesn't know about, so on the next frame it tries to draw with shader 0.
// This leads to problems when invoking GlTextureCallbacks, because those don't expect
// the OpenGL state to change between calls. Also see: https://github.com/servo/webrender/pull/2880
//
// NOTE: For some reason, webrender allows rendering to a framebuffer with a
// negative width / height, although that doesn't make sense
fn render_inner<T>(
    window: &mut Window<T>,
    app_resources: &mut AppResources,
    mut txn: Transaction,
    background_color: ColorU,
) {

    use window::get_gl_context;
    use glium::glutin::ContextTrait;
    use webrender::api::{DeviceIntRect, DeviceIntPoint};
    use azul_css::ColorF;

    let (_, framebuffer_size) = convert_window_size(&window.state.size);

    // Especially during minimization / maximization of a window, it can happen that the window
    // width or height is zero. In that case, no rendering is necessary (doing so would crash
    // the application, since glTexImage2D may never have a 0 as the width or height.
    if framebuffer_size.width == 0 || framebuffer_size.height == 0 {
        return;
    }

    window.internal.epoch = increase_epoch(window.internal.epoch);

    txn.set_window_parameters(
        framebuffer_size.clone(),
        DeviceIntRect::new(DeviceIntPoint::new(0, 0), framebuffer_size),
        window.state.size.hidpi_factor as f32
    );
    txn.set_root_pipeline(window.internal.pipeline_id);
    scroll_all_nodes(&mut window.scroll_states, &mut txn);
    txn.generate_frame();

    app_resources.fake_display.render_api.send_transaction(window.internal.document_id, txn);

    // Update WR texture cache
    app_resources.fake_display.renderer.as_mut().unwrap().update();

    let background_color_f: ColorF = background_color.into();

    unsafe {

        // NOTE: GlContext is the context of the app-global, hidden window
        // (that shares the renderer), not the context of the window itself.
        let gl_context = get_gl_context(&app_resources.fake_display.hidden_display).unwrap();

        // NOTE: The `hidden_display` must share the OpenGL context with the `window`,
        // otherwise this will segfault! Use `ContextBuilder::with_shared_lists` to share the
        // OpenGL context across different windows.
        //
        // The context **must** be made current before calling `.bind_framebuffer()`,
        // otherwise EGL will panic with EGL_BAD_MATCH. The current context has to be the
        // hidden_display context, otherwise this will segfault on Windows.
        app_resources.fake_display.hidden_display.gl_window().make_current().unwrap();

        let mut current_program = [0_i32];
        gl_context.get_integer_v(gl::CURRENT_PROGRAM, &mut current_program);

        // Generate a framebuffer (that will contain the final, rendered screen output).
        let framebuffers = gl_context.gen_framebuffers(1);
        gl_context.bind_framebuffer(gl::FRAMEBUFFER, framebuffers[0]);

        // Create the texture to render to
        let textures = gl_context.gen_textures(1);

        gl_context.bind_texture(gl::TEXTURE_2D, textures[0]);
        gl_context.tex_image_2d(gl::TEXTURE_2D, 0, gl::RGB as i32, framebuffer_size.width, framebuffer_size.height, 0, gl::RGB, gl::UNSIGNED_BYTE, None);

        gl_context.tex_parameter_i(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
        gl_context.tex_parameter_i(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
        gl_context.tex_parameter_i(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl_context.tex_parameter_i(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);

        let depthbuffers = gl_context.gen_renderbuffers(1);
        gl_context.bind_renderbuffer(gl::RENDERBUFFER, depthbuffers[0]);
        gl_context.renderbuffer_storage(gl::RENDERBUFFER, gl::DEPTH_COMPONENT, framebuffer_size.width, framebuffer_size.height);
        gl_context.framebuffer_renderbuffer(gl::FRAMEBUFFER, gl::DEPTH_ATTACHMENT, gl::RENDERBUFFER, depthbuffers[0]);

        // Set "textures[0]" as the color attachement #0
        gl_context.framebuffer_texture_2d(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_2D, textures[0], 0);

        gl_context.draw_buffers(&[gl::COLOR_ATTACHMENT0]);

        // Check that the framebuffer is complete
        assert_eq!(gl_context.check_frame_buffer_status(gl::FRAMEBUFFER), gl::FRAMEBUFFER_COMPLETE);

        // Invoke WebRender to render the frame - renders to the currently bound FB
        gl_context.clear_color(background_color_f.r, background_color_f.g, background_color_f.b, background_color_f.a);
        gl_context.clear_depth(0.0);

        // Disable SRGB and multisample, otherwise, WebRender will crash
        gl_context.disable(gl::FRAMEBUFFER_SRGB);
        gl_context.disable(gl::MULTISAMPLE);
        gl_context.disable(gl::POLYGON_SMOOTH);

        app_resources.fake_display.renderer.as_mut().unwrap().render(framebuffer_size).unwrap();

        gl_context.delete_framebuffers(&framebuffers);
        gl_context.delete_renderbuffers(&depthbuffers);

        // FBOs can't be shared between windows, but textures can.
        // In order to draw on the windows backbuffer, first make the window current, then draw to FB 0
        window.display.gl_window().make_current().unwrap();
        let window_context = get_gl_context(&window.display).unwrap();
        draw_texture_to_screen(&*window_context, textures[0], framebuffer_size);
        window.display.swap_buffers().unwrap();

        app_resources.fake_display.hidden_display.gl_window().make_current().unwrap();

        // Only delete the texture here...
        gl_context.delete_textures(&textures);

        gl_context.bind_framebuffer(gl::FRAMEBUFFER, 0);
        gl_context.bind_texture(gl::TEXTURE_2D, 0);
        gl_context.use_program(current_program[0] as u32);

    };

    // The initial setup can lead to flickering during startup, by default
    // the window is hidden until the first frame has been rendered.
    if window.create_options.state.is_visible && window.state.is_visible {
        window.display.gl_window().window().show();
        window.state.is_visible = true;
        window.create_options.state.is_visible = false;
    }
}

/// When called with glDrawArrays(0, 3), generates a simple triangle that
/// spans the whole screen.
const DISPLAY_VERTEX_SHADER: &[u8] = b"
    #version 140
    out vec2 vTexCoords;
    void main() {
        float x = -1.0 + float((gl_VertexID & 1) << 2);
        float y = -1.0 + float((gl_VertexID & 2) << 1);
        vTexCoords = vec2((x+1.0)*0.5, (y+1.0)*0.5);
        gl_Position = vec4(x, y, 0, 1);
    }
\0";

/// Shader that samples an input texture (`fScreenTex`) to the output FB.
const DISPLAY_FRAGMENT_SHADER: &[u8] = b"
    #version 140
    in vec2 vTexCoords;
    uniform sampler2D fScreenTex;
    out vec4 fColorOut;

    void main() {
        fColorOut = texture(fScreenTex, vTexCoords);
    }
\0";

// NOTE: Compilation is thread-unsafe, should only be compiled on the main thread
static mut DISPLAY_SHADER: Option<GLuint> = None;

/// Compiles the display vertex / fragment shader, returns the compiled shaders.
fn compile_screen_shader(context: &Gl) -> GLuint {

    unsafe {
        match DISPLAY_SHADER {
            Some(s) => return s,
            None => { },
        }
    }

    let vertex_shader_object = context.create_shader(gl::VERTEX_SHADER);
    context.shader_source(vertex_shader_object, &[DISPLAY_VERTEX_SHADER]);
    context.compile_shader(vertex_shader_object);
    if get_gl_shader_error(context, vertex_shader_object) {
        let err = context.get_shader_info_log(vertex_shader_object);
        context.delete_shader(vertex_shader_object);
        panic!("VS compile error: {}", err);
    }

    let fragment_shader_object = context.create_shader(gl::FRAGMENT_SHADER);
    context.shader_source(fragment_shader_object, &[DISPLAY_FRAGMENT_SHADER]);
    context.compile_shader(fragment_shader_object);
    if get_gl_shader_error(context, fragment_shader_object) {
        let err = context.get_shader_info_log(fragment_shader_object);
        context.delete_shader(vertex_shader_object);
        context.delete_shader(fragment_shader_object);
        panic!("FS compile error: {}", err);
    }

    let program = context.create_program();
    context.attach_shader(program, vertex_shader_object);
    context.attach_shader(program, fragment_shader_object);
    context.link_program(program);
    if get_gl_program_error(context, program) {
        let err = context.get_program_info_log(program);
        context.delete_shader(vertex_shader_object);
        context.delete_shader(fragment_shader_object);
        context.delete_program(program);
        panic!("Program link error: {}", err);
    }

    // context.detach_shader(program, vertex_shader_object);
    // context.detach_shader(program, fragment_shader_object);
    context.delete_shader(vertex_shader_object);
    context.delete_shader(fragment_shader_object);

    unsafe { DISPLAY_SHADER = Some(program) };

    program
}

// Returns true on error, false otherwise
fn get_gl_shader_error(context: &Gl, shader_object: GLuint) -> bool {
    let mut err = [0];
    unsafe { context.get_shader_iv(shader_object, gl::COMPILE_STATUS, &mut err) };
    err[0] == 0
}

fn get_gl_program_error(context: &Gl, shader_object: GLuint) -> bool {
    let mut err = [0];
    unsafe { context.get_program_iv(shader_object, gl::LINK_STATUS, &mut err) };
    err[0] == 0
}

// Draws a texture to the currently bound framebuffer. Texture has to be cleaned up by the caller.
fn draw_texture_to_screen(context: &Gl, texture: GLuint, framebuffer_size: DeviceIntSize) {

    context.bind_framebuffer(gl::FRAMEBUFFER, 0);

    // Compile or get the cached shader
    let shader = compile_screen_shader(context);
    let texture_location = context.get_uniform_location(shader, "fScreenTex");

    // The uniform value for a sampler refers to the texture unit, not the texture id, i.e.:
    //
    // TEXTURE0 = uniform_1i(location, 0);
    // TEXTURE1 = uniform_1i(location, 1);

    context.active_texture(gl::TEXTURE0);
    context.bind_texture(gl::TEXTURE_2D, texture);
    context.use_program(shader);
    context.uniform_1i(texture_location, 0);

    // The vertices are generated in the vertex shader using gl_VertexID, however,
    // drawing without a VAO is not allowed (except for glDrawArraysInstanced,
    // which is only available in OGL 3.3)

    let vao = context.gen_vertex_arrays(1);
    context.bind_vertex_array(vao[0]);
    context.viewport(0, 0, framebuffer_size.width, framebuffer_size.height);
    context.draw_arrays(gl::TRIANGLE_STRIP, 0, 3);
    context.delete_vertex_arrays(&vao);

    context.bind_vertex_array(0);
    context.use_program(0);
    context.bind_texture(gl::TEXTURE_2D, 0);
}

fn set_webrender_debug_flags(r: &mut Renderer, old_flags: &DebugState, new_flags: &DebugState) {

    use webrender::DebugFlags;

    if old_flags.profiler_dbg != new_flags.profiler_dbg {
        r.set_debug_flag(DebugFlags::PROFILER_DBG, new_flags.profiler_dbg);
    }
    if old_flags.render_target_dbg != new_flags.render_target_dbg {
        r.set_debug_flag(DebugFlags::RENDER_TARGET_DBG, new_flags.render_target_dbg);
    }
    if old_flags.texture_cache_dbg != new_flags.texture_cache_dbg {
        r.set_debug_flag(DebugFlags::TEXTURE_CACHE_DBG, new_flags.texture_cache_dbg);
    }
    if old_flags.gpu_time_queries != new_flags.gpu_time_queries {
        r.set_debug_flag(DebugFlags::GPU_TIME_QUERIES, new_flags.gpu_time_queries);
    }
    if old_flags.gpu_sample_queries != new_flags.gpu_sample_queries {
        r.set_debug_flag(DebugFlags::GPU_SAMPLE_QUERIES, new_flags.gpu_sample_queries);
    }
    if old_flags.disable_batching != new_flags.disable_batching {
        r.set_debug_flag(DebugFlags::DISABLE_BATCHING, new_flags.disable_batching);
    }
    if old_flags.epochs != new_flags.epochs {
        r.set_debug_flag(DebugFlags::EPOCHS, new_flags.epochs);
    }
    if old_flags.compact_profiler != new_flags.compact_profiler {
        r.set_debug_flag(DebugFlags::COMPACT_PROFILER, new_flags.compact_profiler);
    }
    if old_flags.echo_driver_messages != new_flags.echo_driver_messages {
        r.set_debug_flag(DebugFlags::ECHO_DRIVER_MESSAGES, new_flags.echo_driver_messages);
    }
    if old_flags.new_frame_indicator != new_flags.new_frame_indicator {
        r.set_debug_flag(DebugFlags::NEW_FRAME_INDICATOR, new_flags.new_frame_indicator);
    }
    if old_flags.new_scene_indicator != new_flags.new_scene_indicator {
        r.set_debug_flag(DebugFlags::NEW_SCENE_INDICATOR, new_flags.new_scene_indicator);
    }
    if old_flags.show_overdraw != new_flags.show_overdraw {
        r.set_debug_flag(DebugFlags::SHOW_OVERDRAW, new_flags.show_overdraw);
    }
    if old_flags.gpu_cache_dbg != new_flags.gpu_cache_dbg {
        r.set_debug_flag(DebugFlags::GPU_CACHE_DBG, new_flags.gpu_cache_dbg);
    }
}
