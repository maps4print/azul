extern crate azul_layout;
extern crate azul_core;
extern crate azul_css_parser;

struct Mock { }

fn main() {

    use std::{rc::Rc, collections::BTreeMap};
    use azul_core::{
        app_resources::{
            AppResources, Epoch, FakeRenderApi,
            ImageData, ImageDescriptor, ImageSource, FontSource,
        },
        dom::{DomId, Dom},
        display_list::{SolvedLayout, CachedDisplayList},
        callbacks::PipelineId,
        gl::VirtualGlDriver,
        ui_state::UiState,
        ui_description::UiDescription,
        window::{WindowSize, FullWindowState, LogicalSize},
    };

    fn load_font(_: &FontSource) -> Option<(Vec<u8>, i32)> { None }
    fn load_image(_: &ImageSource) -> Option<(ImageData, ImageDescriptor)> { None }

    let mut data = Mock { };
    let mut app_resources = AppResources::new();
    let mut render_api = FakeRenderApi::new();

    // Set width + height of the rendering here
    let (page_width_px, page_height_px) = (600.0, 100.0);
    // TODO: why &mut?
    let mut fake_window_state = FullWindowState {
        size: WindowSize {
            dimensions: LogicalSize::new(page_width_px, page_height_px),
            .. Default::default()
        },
        .. Default::default()
    };

    let gl_context = Rc::new(VirtualGlDriver::new());
    let pipeline_id = PipelineId::new();
    let epoch = Epoch(0);
    let dom: Dom<Mock> = Dom::div().with_id("hello");
    let css = azul_css_parser::new_from_str("#hello {
        width: 300px;
        height: 40px;
        background: red;
    }").unwrap();

    let mut ui_state = UiState::new(dom, None);
    let ui_description = UiDescription::new(&mut ui_state, &css, &None, &BTreeMap::new(), false);

    let mut ui_states = BTreeMap::new();
    ui_states.insert(DomId::ROOT_ID, ui_state);
    let mut ui_descriptions = BTreeMap::new();
    ui_descriptions.insert(DomId::ROOT_ID, ui_description);
    let mut default_callbacks = BTreeMap::new();

    // Solve the layout (the extra parameters are necessary because of IFrame recursion)
    let solved_layout = SolvedLayout::new(
        &mut data,
        &pipeline_id,
        epoch,
        &mut render_api,
        &mut app_resources,
        gl_context.clone(),
        &mut fake_window_state,
        &mut ui_states,
        &mut ui_descriptions,
        &mut default_callbacks,
        azul_core::gl::insert_into_active_gl_textures,
        azul_layout::ui_solver::do_the_layout,
        load_font,
        load_image,
    );

    let display_list = CachedDisplayList::new(
        epoch,
        pipeline_id,
        &fake_window_state,
        &ui_states,
        &solved_layout.solved_layout_cache,
        &solved_layout.gl_texture_cache,
        &app_resources,
    );

    // Do the rendering for your custom backend here
    println!("{:#?}", display_list);
}
