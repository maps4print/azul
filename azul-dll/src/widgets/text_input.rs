//! Text input (demonstrates two-way data binding)

use core::ops::Range;
use azul::{
    css::*,
    str::String as AzString,
    style::StyledDom,
    vec::{
        NodeDataInlineCssPropertyVec,
        StyleBackgroundContentVec,
        StyleFontFamilyVec,
    },
    dom::{
        Dom, NodeDataInlineCssProperty,
        NodeDataInlineCssProperty::{Normal, Hover, Focus}
    },
    window::{KeyboardState, VirtualKeyCode},
    callbacks::{RefAny, Callback, CallbackInfo, Update},
};
use alloc::vec::Vec;
use alloc::string::String;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct TextInput {
    pub state: TextInputStateWrapper,
    pub placeholder_style: NodeDataInlineCssPropertyVec,
    pub container_style: NodeDataInlineCssPropertyVec,
    pub label_style: NodeDataInlineCssPropertyVec,
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(C)]
pub struct OnTextInputReturn {
    pub update: Update,
    pub valid: TextInputValid,
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(C)]
pub enum TextInputValid {
    Yes,
    No,
}

// The text input field has a special return which specifies
// whether the text input should handle the character
pub type TextInputCallback = extern "C" fn(&mut RefAny, &TextInputState, &mut CallbackInfo) -> OnTextInputReturn;
#[repr(C)]
pub struct TextInputCallbackFn { pub cb: TextInputCallback }
impl_callback!(TextInputCallbackFn);

pub type VirtualKeyDownCallback = extern "C" fn(&mut RefAny, &TextInputState, &mut CallbackInfo) -> OnTextInputReturn;
#[repr(C)]
pub struct VirtualKeyDownCallbackFn { pub cb: VirtualKeyDownCallback }
impl_callback!(VirtualKeyDownCallbackFn);

pub type OnFocusLostCallback = extern "C" fn(&mut RefAny, &TextInputState, &mut CallbackInfo) -> Update;
#[repr(C)]
pub struct OnFocusLostCallbackFn { pub cb: OnFocusLostCallback }
impl_callback!(OnFocusLostCallbackFn);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct TextInputState {
    pub text: CharVec,
    pub placeholder: OptionAzString,
    pub max_len: usize,
    pub selection: OptionTextInputSelection,
    pub cursor_pos: usize,
}

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct TextInputStateWrapper {
    pub inner: TextInputState,
    pub on_text_input: OptionTextInputOnTextInput,
    pub on_virtual_key_down: OptionTextInputOnVirtualKeyDown,
    pub on_focus_lost: OptionTextInputOnFocusLost,
    pub update_text_input_before_calling_focus_lost_fn: bool,
    pub update_text_input_before_calling_vk_down_fn: bool,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[repr(C)]
pub struct TextInputOnTextInput {
    pub callback: TextInputCallbackFn,
    pub data: RefAny,
}

impl_option!(TextInputOnTextInput, OptionTextInputOnTextInput, copy = false, [Debug, Clone, PartialEq, PartialOrd]);

#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[repr(C)]
pub struct TextInputOnVirtualKeyDown {
    pub callback: TextInputCallbackFn,
    pub data: RefAny,
}

impl_option!(TextInputOnVirtualKeyDown, OptionTextInputOnVirtualKeyDown, copy = false, [Debug, Clone, PartialEq, PartialOrd]);

#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[repr(C)]
pub struct TextInputOnFocusLost {
    pub callback: TextInputCallbackFn,
    pub data: RefAny,
}

impl_option!(TextInputOnFocusLost, OptionTextInputOnFocusLost, copy = false, [Debug, Clone, PartialEq, PartialOrd]);

const BACKGROUND_COLOR: ColorU = ColorU { r: 255,  g: 255,  b: 255,  a: 255 }; // white
const TEXT_COLOR: StyleTextColor = StyleTextColor { inner: ColorU { r: 0, g: 0, b: 0, a: 255 } }; // black
const COLOR_9B9B9B: ColorU = ColorU { r: 155, g: 155, b: 155, a: 255 }; // #9b9b9b
const COLOR_4286F4: ColorU = ColorU { r: 66, g: 134, b: 244, a: 255 }; // #4286f4
const COLOR_4C4C4C: ColorU = ColorU { r: 76, g: 76, b: 76, a: 255 }; // #4C4C4C
const BACKGROUND_THEME_LIGHT: &[StyleBackgroundContent] = &[StyleBackgroundContent::Color(BACKGROUND_COLOR)];
const BACKGROUND_COLOR_LIGHT: StyleBackgroundContentVec = StyleBackgroundContentVec::from_const_slice(BACKGROUND_THEME_LIGHT);

const SANS_SERIF_STR: &str = "sans-serif";
const SANS_SERIF: AzString = AzString::from_const_str(SANS_SERIF_STR);
const SANS_SERIF_FAMILIES: &[StyleFontFamily] = &[StyleFontFamily::System(SANS_SERIF)];
const SANS_SERIF_FAMILY: StyleFontFamilyVec = StyleFontFamilyVec::from_const_slice(SANS_SERIF_FAMILIES);

// -- container style

#[cfg(target_os = "windows")]
static TEXT_INPUT_CONTAINER_PROPS: &[NodeDataInlineCssProperty] = &[

    Normal(CssProperty::const_cursor(StyleCursor::Text)),
    Normal(CssProperty::const_box_sizing(LayoutBoxSizing::BorderBox)),
    Normal(CssProperty::const_min_width(LayoutMinWidth::const_px(200))),
    Normal(CssProperty::const_background_content(BACKGROUND_COLOR_LIGHT)),

    Normal(CssProperty::const_padding_left(LayoutPaddingLeft::const_px(2))),
    Normal(CssProperty::const_padding_right(LayoutPaddingRight::const_px(2))),
    Normal(CssProperty::const_padding_top(LayoutPaddingTop::const_px(1))),
    Normal(CssProperty::const_padding_bottom(LayoutPaddingBottom::const_px(1))),

    // border: 1px solid #484c52;

    Normal(CssProperty::const_border_top_width(LayoutBorderTopWidth::const_px(1))),
    Normal(CssProperty::const_border_bottom_width(LayoutBorderBottomWidth::const_px(1))),
    Normal(CssProperty::const_border_left_width(LayoutBorderLeftWidth::const_px(1))),
    Normal(CssProperty::const_border_right_width(LayoutBorderRightWidth::const_px(1))),

    Normal(CssProperty::const_border_top_style(StyleBorderTopStyle { inner: BorderStyle::Inset })),
    Normal(CssProperty::const_border_bottom_style(StyleBorderBottomStyle { inner: BorderStyle::Inset })),
    Normal(CssProperty::const_border_left_style(StyleBorderLeftStyle { inner: BorderStyle::Inset })),
    Normal(CssProperty::const_border_right_style(StyleBorderRightStyle { inner: BorderStyle::Inset })),

    Normal(CssProperty::const_border_top_color(StyleBorderTopColor { inner: COLOR_9B9B9B })),
    Normal(CssProperty::const_border_bottom_color(StyleBorderBottomColor { inner: COLOR_9B9B9B })),
    Normal(CssProperty::const_border_left_color(StyleBorderLeftColor { inner: COLOR_9B9B9B })),
    Normal(CssProperty::const_border_right_color(StyleBorderRightColor { inner: COLOR_9B9B9B })),

    Normal(CssProperty::const_overflow_x(LayoutOverflow::Hidden)),
    Normal(CssProperty::const_overflow_y(LayoutOverflow::Hidden)),
    Normal(CssProperty::const_justify_content(LayoutJustifyContent::Center)),

    // Hover(border-color: #4c4c4c;)

    Hover(CssProperty::const_border_top_color(StyleBorderTopColor { inner: COLOR_4C4C4C })),
    Hover(CssProperty::const_border_bottom_color(StyleBorderBottomColor { inner: COLOR_4C4C4C })),
    Hover(CssProperty::const_border_left_color(StyleBorderLeftColor { inner: COLOR_4C4C4C })),
    Hover(CssProperty::const_border_right_color(StyleBorderRightColor { inner: COLOR_4C4C4C })),

    // Focus(border-color: #4286f4;)

    Focus(CssProperty::const_border_top_color(StyleBorderTopColor { inner: COLOR_4286F4 })),
    Focus(CssProperty::const_border_bottom_color(StyleBorderBottomColor { inner: COLOR_4286F4 })),
    Focus(CssProperty::const_border_left_color(StyleBorderLeftColor { inner: COLOR_4286F4 })),
    Focus(CssProperty::const_border_right_color(StyleBorderRightColor { inner: COLOR_4286F4 })),
];

#[cfg(target_os = "linux")]
static TEXT_INPUT_CONTAINER_PROPS: &[NodeDataInlineCssProperty] = &[

    Normal(CssProperty::const_cursor(StyleCursor::Text)),
    Normal(CssProperty::const_box_sizing(LayoutBoxSizing::BorderBox)),
    Normal(CssProperty::const_font_size(StyleFontSize::const_px(13))),
    Normal(CssProperty::const_min_width(LayoutMinWidth::const_px(200))),
    Normal(CssProperty::const_background_content(BACKGROUND_COLOR_LIGHT)),
    Normal(CssProperty::const_text_color(StyleTextColor { inner: COLOR_4C4C4C })),

    Normal(CssProperty::const_padding_left(LayoutPaddingLeft::const_px(2))),
    Normal(CssProperty::const_padding_right(LayoutPaddingRight::const_px(2))),
    Normal(CssProperty::const_padding_top(LayoutPaddingTop::const_px(1))),
    Normal(CssProperty::const_padding_bottom(LayoutPaddingBottom::const_px(1))),

    // border: 1px solid #484c52;

    Normal(CssProperty::const_border_top_width(LayoutBorderTopWidth::const_px(1))),
    Normal(CssProperty::const_border_bottom_width(LayoutBorderBottomWidth::const_px(1))),
    Normal(CssProperty::const_border_left_width(LayoutBorderLeftWidth::const_px(1))),
    Normal(CssProperty::const_border_right_width(LayoutBorderRightWidth::const_px(1))),

    Normal(CssProperty::const_border_top_style(StyleBorderTopStyle { inner: BorderStyle::Inset })),
    Normal(CssProperty::const_border_bottom_style(StyleBorderBottomStyle { inner: BorderStyle::Inset })),
    Normal(CssProperty::const_border_left_style(StyleBorderLeftStyle { inner: BorderStyle::Inset })),
    Normal(CssProperty::const_border_right_style(StyleBorderRightStyle { inner: BorderStyle::Inset })),

    Normal(CssProperty::const_border_top_color(StyleBorderTopColor { inner: COLOR_9B9B9B })),
    Normal(CssProperty::const_border_bottom_color(StyleBorderBottomColor { inner: COLOR_9B9B9B })),
    Normal(CssProperty::const_border_left_color(StyleBorderLeftColor { inner: COLOR_9B9B9B })),
    Normal(CssProperty::const_border_right_color(StyleBorderRightColor { inner: COLOR_9B9B9B })),

    Normal(CssProperty::const_overflow_x(LayoutOverflow::Hidden)),
    Normal(CssProperty::const_overflow_y(LayoutOverflow::Hidden)),
    Normal(CssProperty::const_text_align(StyleTextAlign::Left)),
    Normal(CssProperty::const_font_size(StyleFontSize::const_px(13))),
    Normal(CssProperty::const_justify_content(LayoutJustifyContent::Center)),

    Normal(CssProperty::const_font_family(SANS_SERIF_FAMILY)),

    // Hover(border-color: #4286f4;)

    Hover(CssProperty::const_border_top_color(StyleBorderTopColor { inner: COLOR_4286F4 })),
    Hover(CssProperty::const_border_bottom_color(StyleBorderBottomColor { inner: COLOR_4286F4 })),
    Hover(CssProperty::const_border_left_color(StyleBorderLeftColor { inner: COLOR_4286F4 })),
    Hover(CssProperty::const_border_right_color(StyleBorderRightColor { inner: COLOR_4286F4 })),

    // Focus(border-color: #4286f4;)

    Focus(CssProperty::const_border_top_color(StyleBorderTopColor { inner: COLOR_4286F4 })),
    Focus(CssProperty::const_border_bottom_color(StyleBorderBottomColor { inner: COLOR_4286F4 })),
    Focus(CssProperty::const_border_left_color(StyleBorderLeftColor { inner: COLOR_4286F4 })),
    Focus(CssProperty::const_border_right_color(StyleBorderRightColor { inner: COLOR_4286F4 })),
];

#[cfg(target_os = "macos")]
static TEXT_INPUT_CONTAINER_PROPS: &[NodeDataInlineCssProperty] = &[

    Normal(CssProperty::const_cursor(StyleCursor::Text)),
    Normal(CssProperty::const_box_sizing(LayoutBoxSizing::BorderBox)),
    Normal(CssProperty::const_min_width(LayoutMinWidth::const_px(200))),
    Normal(CssProperty::const_background_content(BACKGROUND_COLOR_LIGHT)),

    Normal(CssProperty::const_padding_left(LayoutPaddingLeft::const_px(2))),
    Normal(CssProperty::const_padding_right(LayoutPaddingRight::const_px(2))),
    Normal(CssProperty::const_padding_top(LayoutPaddingTop::const_px(1))),
    Normal(CssProperty::const_padding_bottom(LayoutPaddingBottom::const_px(1))),

    // border: 1px solid #484c52;

    Normal(CssProperty::const_border_top_width(LayoutBorderTopWidth::const_px(1))),
    Normal(CssProperty::const_border_bottom_width(LayoutBorderBottomWidth::const_px(1))),
    Normal(CssProperty::const_border_left_width(LayoutBorderLeftWidth::const_px(1))),
    Normal(CssProperty::const_border_right_width(LayoutBorderRightWidth::const_px(1))),

    Normal(CssProperty::const_border_top_style(StyleBorderTopStyle { inner: BorderStyle::Inset })),
    Normal(CssProperty::const_border_bottom_style(StyleBorderBottomStyle { inner: BorderStyle::Inset })),
    Normal(CssProperty::const_border_left_style(StyleBorderLeftStyle { inner: BorderStyle::Inset })),
    Normal(CssProperty::const_border_right_style(StyleBorderRightStyle { inner: BorderStyle::Inset })),

    Normal(CssProperty::const_border_top_color(StyleBorderTopColor { inner: COLOR_9B9B9B })),
    Normal(CssProperty::const_border_bottom_color(StyleBorderBottomColor { inner: COLOR_9B9B9B })),
    Normal(CssProperty::const_border_left_color(StyleBorderLeftColor { inner: COLOR_9B9B9B })),
    Normal(CssProperty::const_border_right_color(StyleBorderRightColor { inner: COLOR_9B9B9B })),

    Normal(CssProperty::const_overflow_x(LayoutOverflow::Hidden)),
    Normal(CssProperty::const_overflow_y(LayoutOverflow::Hidden)),
    Normal(CssProperty::const_text_align(StyleTextAlign::Left)),
    Normal(CssProperty::const_justify_content(LayoutJustifyContent::Center)),

    // Hover(border-color: #4286f4;)

    Hover(CssProperty::const_border_top_color(StyleBorderTopColor { inner: COLOR_4286F4 })),
    Hover(CssProperty::const_border_bottom_color(StyleBorderBottomColor { inner: COLOR_4286F4 })),
    Hover(CssProperty::const_border_left_color(StyleBorderLeftColor { inner: COLOR_4286F4 })),
    Hover(CssProperty::const_border_right_color(StyleBorderRightColor { inner: COLOR_4286F4 })),

    // Focus(border-color: #4286f4;)

    Focus(CssProperty::const_border_top_color(StyleBorderTopColor { inner: COLOR_4286F4 })),
    Focus(CssProperty::const_border_bottom_color(StyleBorderBottomColor { inner: COLOR_4286F4 })),
    Focus(CssProperty::const_border_left_color(StyleBorderLeftColor { inner: COLOR_4286F4 })),
    Focus(CssProperty::const_border_right_color(StyleBorderRightColor { inner: COLOR_4286F4 })),
];

// -- label style

#[cfg(target_os = "windows")]
static TEXT_INPUT_LABEL_PROPS: &[NodeDataInlineCssProperty] = &[
    Normal(CssProperty::const_font_size(StyleFontSize::const_px(13))),
    Normal(CssProperty::const_text_color(StyleTextColor { inner: COLOR_4C4C4C })),
    Normal(CssProperty::const_font_family(SANS_SERIF_FAMILY)),
];

#[cfg(target_os = "linux")]
static TEXT_INPUT_LABEL_PROPS: &[NodeDataInlineCssProperty] = &[
    Normal(CssProperty::const_font_size(StyleFontSize::const_px(13))),
    Normal(CssProperty::const_text_color(StyleTextColor { inner: COLOR_4C4C4C })),
    Normal(CssProperty::const_font_family(SANS_SERIF_FAMILY)),
];

#[cfg(target_os = "macos")]
static TEXT_INPUT_LABEL_PROPS: &[NodeDataInlineCssProperty] = &[
    Normal(CssProperty::const_font_size(StyleFontSize::const_px(13))),
    Normal(CssProperty::const_text_color(StyleTextColor { inner: COLOR_4C4C4C })),
    Normal(CssProperty::const_font_family(SANS_SERIF_FAMILY)),
];

impl Default for TextInput {
    fn default() -> Self {
        TextInput {
            state: TextInputStateWrapper::default(),
            placeholder_style: Vec::new().into(), // TEXT_INPUT_PLACEHOLDER_PROPS
            container_style: NodeDataInlineCssPropertyVec::from_const_slice(TEXT_INPUT_CONTAINER_PROPS),
            label_style: NodeDataInlineCssPropertyVec::from_const_slice(TEXT_INPUT_LABEL_PROPS),
        }
    }
}

impl Default for TextInputState {
    fn default() -> Self {
        TextInputState {
            text: Vec::new().into(),
            placeholder: None.into(),
            max_len: 50,
            selection: None.into(),
            cursor_pos: 0,
        }
    }
}

impl Default for TextInputStateWrapper {
    fn default() -> Self {
        TextInputStateWrapper {
            inner: TextInputState::default(),
            on_text_input: None.into(),
            on_virtual_key_down: None.into(),
            on_focus_lost: None.into(),
            update_text_input_before_calling_focus_lost_fn: true,
            update_text_input_before_calling_vk_down_fn: true,
        }
    }
}

impl TextInput {

    pub fn new(s: AzString) -> Self {
        Self {
            state: TextInputStateWrapper {
                inner: TextInputState {
                    text: s.as_ref().chars().collect(),
                    .. Default::default()
                },
                .. Default::default()
            },
            .. Default::default()
        }
    }

    pub fn set_on_text_input(mut self, callback: TextInputCallback, data: RefAny) {
        self.state.on_text_input = Some((TextInputCallbackFn { cb: callback }, data)).into();
    }

    pub fn set_on_virtual_key_down(mut self, callback: VirtualKeyDownCallback, data: RefAny) {
        self.state.on_virtual_key_down = Some((VirtualKeyDownCallbackFn { cb: callback }, data)).into();
    }

    pub fn set_on_focus_lost(mut self, callback: OnFocusLostCallback, data: RefAny) {
        self.state.on_focus_lost = Some((OnFocusLostCallbackFn { cb: callback }, data)).into();
    }

    pub fn set_placeholder_style(mut self, style: NodeDataInlineCssPropertyVec) {
        self.placeholder_style = style;
    }

    pub fn set_container_style(mut self, style: NodeDataInlineCssPropertyVec) {
        self.container_style = style;
    }

    pub fn set_label_style(mut self, style: NodeDataInlineCssPropertyVec) {
        self.label_style = style;
    }

    pub fn dom(self) -> Dom {

        use azul::dom::{
            CallbackData, EventFilter,
            HoverEventFilter, FocusEventFilter,
            IdOrClass::Class, TabIndex,
        };

        let label_text = self.state.inner.text.iter().collect::<String>();
        let state_ref = RefAny::new(self.state);

        Dom::div()
        .with_ids_and_classes(vec![Class("__azul-native-text-input-container".into())].into())
        .with_inline_css_props(self.container_style)
        .with_tab_index(TabIndex::Auto)
        .with_dataset(state_ref.clone())
        .with_callbacks(vec![
            CallbackData {
                event: EventFilter::Focus(FocusEventFilter::FocusReceived),
                data: state_ref.clone(),
                callback: Callback { cb: self::input::default_on_focus_received }
            },
            CallbackData {
                event: EventFilter::Focus(FocusEventFilter::FocusLost),
                data: state_ref.clone(),
                callback: Callback { cb: self::input::default_on_focus_lost }
            },
            CallbackData {
                event: EventFilter::Focus(FocusEventFilter::TextInput),
                data: state_ref.clone(),
                callback: Callback { cb: self::input::default_on_text_input }
            },
            CallbackData {
                event: EventFilter::Focus(FocusEventFilter::VirtualKeyDown),
                data: state_ref.clone(),
                callback: Callback { cb: self::input::default_on_virtual_key_down }
            },
            CallbackData {
                event: EventFilter::Hover(HoverEventFilter::LeftMouseDown),
                data: state_ref.clone(),
                callback: Callback { cb: self::input::default_on_container_click }
            },
        ].into())
        .with_children(vec![
            Dom::text(label_text.into())
            .with_ids_and_classes(vec![Class("__azul-native-text-input-label".into())].into())
            .with_inline_css_props(self.label_style),
            // let cursor = Dom::div().with_class("__azul-native-text-input-cursor");
            // let text_selection = Dom::div().with_class("__azul-native-text-input-selection".into());
        ].into())
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
#[repr(C, u8)]
pub enum TextInputSelection {
    All,
    FromTo(TextInputSelectionRange),
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct TextInputSelectionRange {
    pub from: usize,
    pub to: usize,
}

impl TextInputSelection {
    pub fn get_range(&self, text_len: usize) -> Range<usize> {
        match self {
            TextInputSelection::All => 0..text_len,
            TextInputSelection::FromTo(r) => r.from..r.to,
        }
    }
}

impl TextInputState {

    pub fn get_text(&self) -> String {
        self.text.clone().into_iter().collect()
    }

    fn handle_on_text_input(&mut self, c: char) {
        match self.selection.clone() {
            None => {
                if self.cursor_pos == self.text.len() {
                    self.text.push(c);
                } else {
                    // TODO: insert character at the cursor location!
                    self.text.insert(self.cursor_pos, c);
                }
                self.cursor_pos = self.cursor_pos.saturating_add(1).min(self.text.len() - 1);
            },
            Some(TextInputSelection::All) => {
                self.text = vec![c];
                self.cursor_pos = 1;
                self.selection = None;
            },
            Some(TextInputSelection::FromTo(range)) => {
                self.delete_selection(range, Some(c));
            },
        }
    }

    fn handle_on_virtual_key_down(
        &mut self,
        virtual_key: VirtualKeyCode,
        keyboard_state: &KeyboardState,
        info: &mut CallbackInfo,
    ) {
        match virtual_key {
            VirtualKeyCode::Back => {
                // TODO: shift + back = delete last word
                let selection = self.selection.clone();
                match selection {
                    None => {
                        if self.cursor_pos == (self.text.len() - 1) {
                            self.text.pop();
                        } else {
                            self.text.remove(self.cursor_pos);
                        }
                        self.cursor_pos = self.cursor_pos.saturating_sub(1);
                    },
                    Some(TextInputSelection::All) => {
                        self.text = Vec::new();
                        self.cursor_pos = 0;
                        self.selection = None;
                    },
                    Some(TextInputSelection::FromTo(range)) => {
                        self.delete_selection(range, None);
                    },
                }
            },
            VirtualKeyCode::Return => { /* ignore return keys */ },
            VirtualKeyCode::Home => {
                self.cursor_pos = 0;
                self.selection = None;
            },
            VirtualKeyCode::End => {
                self.cursor_pos = self.text.len().saturating_sub(1);
                self.selection = None;
            },
            VirtualKeyCode::Tab => {
                use azul::callbacks::FocusTarget;
                if keyboard_state.shift_down {
                    info.set_focus(FocusTarget::Next);
                } else {
                    info.set_focus(FocusTarget::Previous);
                }
            },
            VirtualKeyCode::Escape => {
                self.selection = None;
            },
            VirtualKeyCode::Right => {
                self.cursor_pos = self.cursor_pos.saturating_add(1).min(self.text.len());
            },
            VirtualKeyCode::Left => {
                self.cursor_pos = self.cursor_pos.saturating_sub(1).min(self.text.len());
            },
            // ctrl + a
            VirtualKeyCode::A if keyboard_state.ctrl_down => {
                self.selection = Some(TextInputSelection::All);
            },
            /*
            // ctrl + c
            VirtualKeyCode::C if keyboard_state.ctrl_down => {
                Clipboard::new().set_string_contents(self.text[self.selection]);
            },
            // ctrl + v
            VirtualKeyCode::V if keyboard_state.ctrl_down => {
                let clipboard_contents = Clipboard::new().get_string_contents(self.text[self.selection]).into_option().unwrap_or_default();
                let clipboard_contents: Vec<char> = clipboard_contents.as_str().chars().collect();
                match self.selection {
                    None => {

                    },
                    Some(TextInputSelection::All) => {
                        self.text = ;
                    },
                    Some(TextInputSelection::Range(r)) => {
                        self.delete_selection(r);


                    }
                }
                self.selection = None;
                self.cursor = ...;
            },
            // ctrl + x
            VirtualKeyCode::X if keyboard_state.ctrl_down => {
                Clipboard::new().set_string_contents(self.text[self.selection.get_range(self.text.len())]);
            }
            */
            _ => { },
        }
    }

    fn delete_selection(&mut self, selection: Range<usize>, new_text: Option<char>) {
        let Range { start, end } = selection;
        let max = end.min(self.text.len() - 1);

        if max == (self.text.len() - 1) {
            self.text.truncate(start);
            if let Some(new) = new_text {
                self.text.push(new);
                self.cursor_pos = start + 1;
            } else {
                self.cursor_pos = start;
            }
        } else {
            let end = &self.text[max..].to_vec();
            self.text.truncate(start);
            if let Some(new) = new_text {
                self.text.push(new);
                self.cursor_pos = start + 1;
            } else {
                self.cursor_pos = start;
            }
            self.text.extend(end.iter());
        }
    }
}

// handle input events for the TextInput
mod input {

    use azul::callbacks::{RefAny, CallbackInfo, Update};
    use super::{TextInputStateWrapper, OnTextInputReturn, TextInputValid};
    use alloc::string::String;

    pub(in super) extern "C" fn default_on_text_input(text_input: &mut RefAny, mut info: CallbackInfo) -> Update {

        let mut text_input = match text_input.downcast_mut::<TextInputStateWrapper>() {
            Some(s) => s,
            None => return Update::DoNothing,
        };

        let keyboard_state = info.get_current_keyboard_state();

        let c = match keyboard_state.current_char.into_option() {
            Some(s) => s,
            None => return Update::DoNothing,
        };

        let c = match core::char::from_u32(c) {
            Some(s) => s,
            None => return Update::DoNothing,
        };

        let label_node_id = match info.get_first_child(info.get_hit_node()).into_option() {
            Some(s) => s,
            None => return Update::DoNothing,
        };

        let result = {
            // rustc doesn't understand the borrowing lifetime here
            let text_input = &mut *text_input;
            let ontextinput = &mut text_input.on_text_input;

            // inner_clone has the new text
            let mut inner_clone = text_input.inner.clone();
            inner_clone.handle_on_text_input(c);

            match ontextinput.as_mut() {
                Some((f, d)) => (f.cb)(d, &inner_clone, &mut info),
                None => OnTextInputReturn {
                    update: Update::DoNothing,
                    valid: TextInputValid::Yes,
                },
            }
        };

        if result.valid == TextInputValid::Yes {
            text_input.inner.handle_on_text_input(c);
        }

        // Update the string, cursor position on the screen and selection background
        // TODO: restart the timer for cursor blinking
        info.set_string_contents(label_node_id, text_input.inner.text.iter().collect::<String>().into());
        // info.set_css_property(cursor_node_id, CssProperty::const_transform(get_cursor_transform(info.get_text_contents()[self.cursor_pos])))
        // info.update_image(selection_node_id, render_selection(self.selection));

        result.update
    }

    pub(in super) extern "C" fn default_on_virtual_key_down(text_input: &mut RefAny, mut info: CallbackInfo) -> Update {

        let mut text_input = match text_input.downcast_mut::<TextInputStateWrapper>() {
            Some(s) => s,
            None => return Update::DoNothing,
        };
        let keyboard_state = info.get_current_keyboard_state();
        let last_keycode = match keyboard_state.current_virtual_keycode.into_option() {
            Some(s) => s,
            None => return Update::DoNothing,
        };

        let kb_state = info.get_current_keyboard_state();

        let result = {
            // rustc doesn't understand the borrowing lifetime here
            let text_input = &mut *text_input;
            let ontextinput = &mut text_input.on_virtual_key_down;

            // inner_clone has the new text
            let mut inner_clone = text_input.inner.clone();
            inner_clone.handle_on_virtual_key_down(last_keycode, &kb_state, &mut info);

            match ontextinput.as_mut() {
                Some((f, d)) => (f.cb)(d, &inner_clone, &mut info),
                None => OnTextInputReturn {
                    update: Update::DoNothing,
                    valid: TextInputValid::Yes,
                },
            }
        };

        if result.valid == TextInputValid::Yes {
            text_input.inner.handle_on_virtual_key_down(last_keycode, &kb_state, &mut info);
        }

        result.update
    }

    pub(in super) extern "C" fn default_on_container_click(text_input: &mut RefAny, info: CallbackInfo) -> Update {
        let mut text_input = match text_input.downcast_mut::<TextInputStateWrapper>() {
            Some(s) => s,
            None => return Update::DoNothing,
        };
        // TODO: clear selection, set cursor to text hit
        Update::DoNothing
    }

    pub(in super) extern "C" fn default_on_label_click(text_input: &mut RefAny, info: CallbackInfo) -> Update {
        let mut text_input = match text_input.downcast_mut::<TextInputStateWrapper>() {
            Some(s) => s,
            None => return Update::DoNothing,
        };
        // TODO: set cursor to end or start
        Update::DoNothing
    }

    pub(in super) extern "C" fn default_on_focus_received(text_input: &mut RefAny, info: CallbackInfo) -> Update {
        let mut text_input = match text_input.downcast_mut::<TextInputStateWrapper>() {
            Some(s) => s,
            None => return Update::DoNothing,
        };

        // TODO: start text cursor blinking
        Update::DoNothing
    }

    pub(in super) extern "C" fn default_on_focus_lost(text_input: &mut RefAny, mut info: CallbackInfo) -> Update {
        let mut text_input = match text_input.downcast_mut::<TextInputStateWrapper>() {
            Some(s) => s,
            None => return Update::DoNothing,
        };

        let result = {
            // rustc doesn't understand the borrowing lifetime here
            let text_input = &mut *text_input;
            let ontextinput = &mut text_input.on_focus_lost;
            let inner = &text_input.inner;

            match ontextinput.as_mut() {
                Some((f, d)) => (f.cb)(d, &inner, &mut info),
                None => Update::DoNothing,
            }
        };

        result
    }
}

impl From<TextInput> for Dom {
    fn from(t: TextInput) -> Dom {
        t.dom()
    }
}