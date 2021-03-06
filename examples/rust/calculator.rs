#![windows_subsystem = "windows"]

use azul::prelude::*;

#[derive(Default)]
pub struct Calculator {
    pub current_operator: Option<OperandStack>,
    pub current_operand_stack: OperandStack,
    pub division_by_zero: bool,
    pub expression: String,
    pub last_event: Option<Event>,
}

#[derive(Clone, Debug)]
pub enum Event {
    Clear,
    InvertSign,
    Percent,
    Divide,
    Multiply,
    Subtract,
    Plus,
    EqualSign,
    Dot,
    Number(u8),
}

pub mod resources {
    macro_rules! FONT_PATH {() => { concat!(env!("CARGO_MANIFEST_DIR"), "/examples/assets/fonts/KoHo-Light.ttf")};}

    static FONT: &[u8] = include_bytes!(FONT_PATH!());
    static CSS: &str = "
        * {
            font-size: 27px;
            font-family: \"KoHo-Light\";
            flex-direction: column;
            box-sizing: border-box;
            flex-grow: 1;
        }

        #expression {
            max-height: 50pt;
            background-color: #444;
            color: white;
            flex-direction: row;
            text-align: right;
            padding-right: 40pt;
            justify-content: flex-end;
        }

        #result {
            max-height: 81pt;
            background: linear-gradient(to top, #111, #444);
            color: white;
            flex-direction: row;
            text-align: right;
            padding-right: 16pt;
            justify-content: flex-end;
            font-size: 60px;
        }

        #numpad-container {
            background-color: #d6d6d6;
        }

        .numpad-button {
            border-right: 1px solid #8d8d8d;
        }

        .row {
            flex-direction: row;
            border-bottom: 1px solid #8d8d8d;
            height: 78px;
        }

        .orange {
            background: linear-gradient(to bottom, #f69135, #f37335);
            color: white;
            border-bottom: 1px solid #8d8d8d;
            width: 98px;
        }

        .orange:focus {
            border: 3px solid blue;
        }

        #zero {
            flex-grow: 2;
            border-bottom: none;
        }
    ";
}

/// Handles UI rendering and callback definition
pub mod ui {

    pub extern "C" fn layout(data: &mut RefAny, _info: LayoutCallbackInfo) -> StyledDom {

        let result = if self.division_by_zero {
            format!("Cannot divide by zero.")
        } else {
            self.current_operand_stack.get_display()
        };

        Dom::div()
            .with_child(Dom::label(self.expression.to_string()).with_id("expression"))
            .with_child(Dom::label(result).with_id("result"))
            .with_child(
                Dom::div()
                    .with_id("numpad-container")
                    .with_child(render_row(["C", "+/-", "%", "/"]))
                    .with_child(render_row(["7", "8", "9", "x"]))
                    .with_child(render_row(["4", "5", "6", "-"]))
                    .with_child(render_row(["1", "2", "3", "+"]))
                    .with_child(
                        Dom::div()
                            .with_class("row")
                            .with_child(numpad_btn("0", "numpad-button").with_id("zero"))
                            .with_child(numpad_btn(".", "numpad-button"))
                            .with_child(numpad_btn("=", "orange")),
                    ),
            )
            .with_callback(EventFilter::Window(WindowEventFilter::TextInput), handle_text_input)
            .with_callback(EventFilter::Window(WindowEventFilter::VirtualKeyDown), handle_virtual_key_input)
    }

    #[inline]
    fn render_row(labels: [&'static str; 4]) -> StyledDom {
        Dom::div()
            .with_class("row")
            .with_child(numpad_btn(labels[0], "numpad-button"))
            .with_child(numpad_btn(labels[1], "numpad-button"))
            .with_child(numpad_btn(labels[2], "numpad-button"))
            .with_child(numpad_btn(labels[3], "orange"))
    }

    #[inline]
    fn numpad_btn(label: &'static str, class: &'static str) -> StyledDom {
        Dom::label(label)
            .with_class(class)
            .with_tab_index(TabIndex::Auto)
            .with_callback(On::MouseUp, handle_mouseclick_numpad_btn)
    }

    extern "C" fn handle_mouseclick_numpad_btn(data: &mut RefAny, info: CallbackInfo) -> UpdateScreen {

        // Figure out which row and column was clicked...
        let (clicked_col_idx, clicked_row_idx) = {
            let mut row_iter = info.parent_nodes();
            row_iter.next()?;
            (info.target_index_in_parent()?, row_iter.current_index_in_parent()?)
        };

        // Figure out what button was clicked from the given row and column, filter bad events
        let event = match (clicked_row_idx, clicked_col_idx) {
            (0, 0) => Event::Clear,
            (0, 1) => Event::InvertSign,
            (0, 2) => Event::Percent,
            (0, 3) => Event::Divide,

            (1, 0) => Event::Number(7),
            (1, 1) => Event::Number(8),
            (1, 2) => Event::Number(9),
            (1, 3) => Event::Multiply,

            (2, 0) => Event::Number(4),
            (2, 1) => Event::Number(5),
            (2, 2) => Event::Number(6),
            (2, 3) => Event::Subtract,

            (3, 0) => Event::Number(1),
            (3, 1) => Event::Number(2),
            (3, 2) => Event::Number(3),
            (3, 3) => Event::Plus,

            (4, 0) => Event::Number(0),
            (4, 1) => Event::Dot,
            (4, 2) => Event::EqualSign,

            _ => return DoNothing, // invalid item
        };

        println!("Got event via mouse input: {:?}", event);
        process_event(info.state, event)
    }

    extern "C" fn handle_text_input(data: &mut RefAny, info: CallbackInfo) -> UpdateScreen {
        let current_key = info.get_keyboard_state().current_char?;
        let event = match current_key {
            '0' => Event::Number(0),
            '1' => Event::Number(1),
            '2' => Event::Number(2),
            '3' => Event::Number(3),
            '4' => Event::Number(4),
            '5' => Event::Number(5),
            '6' => Event::Number(6),
            '7' => Event::Number(7),
            '8' => Event::Number(8),
            '9' => Event::Number(9),
            '*' => Event::Multiply,
            '-' => Event::Subtract,
            '+' => Event::Plus,
            '/' => Event::Divide,
            '%' => Event::Percent,
            '.' | ',' => Event::Dot,
            _ => return DoNothing,
        };

        println!("Got event via keyboard input: {:?}", event);
        process_event(info.state, event)
    }

    extern "C" fn handle_virtual_key_input(data: &mut RefAny, info: CallbackInfo) -> UpdateScreen {
        let current_key = info.get_keyboard_state().current_virtual_keycode?;
        let event = match current_key {
            VirtualKeyCode::Return => Event::EqualSign,
            VirtualKeyCode::Back => Event::Clear,
            _ => return DoNothing,
        };
        process_event(info.state, event)
    }
}

/// Handles the application logic
pub mod logic {

    #[derive(Debug, Clone, Default)]
    pub struct OperandStack {
        pub stack: Vec<Number>,
        pub negative_number: bool,
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    enum Number {
        Value(u8),
        Dot,
    }

    impl OperandStack {
        /// Returns the displayable string, i.e for:
        /// `[3, 4, Dot, 5]` => `"34.5"`
        pub fn get_display(&self) -> String {
            let mut display_string = String::new();

            if self.negative_number {
                display_string.push('-');
            }

            if self.stack.is_empty() {
                display_string.push('0');
            } else {
                // If we get a dot at the end of the stack, i.e. "35." - store it,
                // but don't display it
                let mut first_dot_found = false;
                for num in &self.stack {
                    match num {
                        Number::Value(v) => display_string.push((v + 48) as char),
                        Number::Dot => {
                            if !first_dot_found {
                                display_string.push('.');
                                first_dot_found = true;
                            }
                        }
                    }
                }
            }

            display_string
        }

        /// Returns the number which you can use to calculate things with
        pub fn get_number(&self) -> f32 {
            let stack_size = self.stack.len();
            if stack_size == 0 {
                return 0.0;
            }

            // Iterate the stack until the first Dot is found
            let first_dot_position = self.stack.iter()
                .position(|x| *x == Number::Dot)
                .and_then(|x| Some(x - 1))
                .unwrap_or(stack_size - 1) as i32;

            let mut final_number = 0.0;

            for (number_position, number) in self.stack.iter().filter_map(|x| match x {
                    Number::Dot => None,
                    Number::Value(v) => Some(v),
                })
                .enumerate()
            {
                // i.e. the 5 in 5432.1 has a distance of 3 to the first dot (meaning 3 zeros)
                let diff_to_first_dot = first_dot_position - number_position as i32;
                final_number += (*number as f32) * 10.0_f32.powi(diff_to_first_dot);
            }

            if self.negative_number {
                final_number = -final_number;
            }
            final_number
        }

        fn from_f32(value: f32) -> Self {
            let mut result = OperandStack::default();
            for c in value.to_string().chars() {
                if c == '-' {
                    result.negative_number = true;
                } else if c == '.' {
                    result.stack.push(Number::Dot);
                } else {
                    result.stack.push(Number::Value((c as u8 - 48) as u8))
                }
            }
            result
        }
    }

    impl Calculator {

        /// Act on the event accordingly
        fn process_event(&mut self, event: Event) -> UpdateScreen {
            match event {
                Event::Clear => {
                    *calculator = Calculator::default();
                    RefreshDom
                }
                Event::InvertSign => {
                    if !calculator.division_by_zero {
                        calculator.current_operand_stack.negative_number = !calculator.current_operand_stack.negative_number;
                    }
                    RefreshDom
                }
                Event::Percent => {

                    if calculator.division_by_zero {
                        return DoNothing;
                    }

                    if let Some(operation) = &calculator.last_event.clone() {
                        if let Some(operand) = calculator.current_operator.clone() {
                            let num = calculator.current_operand_stack.get_number();
                            let op = operand.get_number();
                            let result = match operation {
                                Event::Plus | Event::Subtract => op / 100.0 * num,
                                Event::Multiply | Event::Divide => num / 100.0,
                                _ => unreachable!(),
                            };
                            calculator.current_operand_stack = OperandStack::from(result);
                        }
                    }

                    RefreshDom
                }
                Event::EqualSign => {

                    if calculator.division_by_zero {
                        return DoNothing;
                    }

                    if let Some(Event::EqualSign) = calculator.last_event {
                        calculator.expression = format!("{} =", calculator.current_operand_stack.get_display());
                    } else {
                        calculator.expression.push_str(&format!("{} =", calculator.current_operand_stack.get_display()));
                        if let Some(operation) = &calculator.last_event.clone() {
                            if let Some(operand) = calculator.current_operator.clone() {
                                let num = calculator.current_operand_stack.get_number();
                                let op = operand.get_number();
                                match perform_operation(op, &operation, num) {
                                    Some(r) => calculator.current_operand_stack = OperandStack::from(r),
                                    None => calculator.division_by_zero = true,
                                }
                            }
                        }
                    }

                    calculator.current_operator = None;
                    calculator.last_event = Some(Event::EqualSign);

                    RefreshDom
                }
                Event::Dot => {

                    if calculator.division_by_zero {
                        return DoNothing;
                    }

                    if calculator.current_operand_stack.stack.iter().position(|x| *x == Number::Dot).is_none() {
                        if calculator.current_operand_stack.stack.len() == 0 {
                            calculator.current_operand_stack.stack.push(Number::Value(0));
                        }
                        calculator.current_operand_stack.stack.push(Number::Dot);
                    }

                    RefreshDom
                }
                Event::Number(v) => {
                    if let Some(Event::EqualSign) = calculator.last_event {
                        *calculator = Calculator::default();
                    }
                    calculator.current_operand_stack.stack.push(Number::Value(v));
                    RefreshDom
                }
                operation => {

                    if calculator.division_by_zero {
                        return DoNothing;
                    }

                    if let Some(Event::EqualSign) = calculator.last_event {
                        calculator.expression = String::new();
                    }

                    calculator.expression.push_str(&calculator.current_operand_stack.get_display());

                    if let Some(Event::EqualSign) = calculator.last_event {
                        calculator.current_operator = Some(calculator.current_operand_stack.clone());
                    } else if let Some(last_operation) = &calculator.last_event.clone() {
                        if let Some(operand) = calculator.current_operator.clone() {
                            let num = calculator.current_operand_stack.get_number();
                            let op = operand.get_number();
                            match perform_operation(op, last_operation, num) {
                                Some(r) => calculator.current_operator = Some(OperandStack::from(r)),
                                None => calculator.division_by_zero = true,
                            }
                        }
                    } else {
                        calculator.current_operator = Some(calculator.current_operand_stack.clone());
                    }

                    calculator.current_operand_stack = OperandStack::default();
                    calculator.expression.push_str(match operation {
                        Event::Plus => " + ",
                        Event::Subtract => " - ",
                        Event::Multiply => " x ",
                        Event::Divide => " / ",
                        _ => unreachable!(),
                    });
                    calculator.last_event = Some(operation);

                    RefreshDom
                }
            }
        }
    }

    impl Event {
        /// Performs an arithmetic operation. Returns None when trying to divide by zero.
        fn perform_operation(left_operand: f32, right_operand: f32) -> Option<f32> {
            match operation {
                Event::Multiply => Some(left_operand * right_operand),
                Event::Subtract => Some(left_operand - right_operand),
                Event::Plus => Some(left_operand + right_operand),
                Event::Divide => if right_operand == 0.0 {
                        None
                    }
                    else {
                        Some(left_operand / right_operand)
                },
                _ => unreachable!(),
            }
        }
    }
}

fn main() {
    let mut app = App::new(Calculator::default(), AppConfig::new(LayoutSolver::Default)).unwrap();
    app.add_font("KoHo-Light", FontRef::embedded(ui::FONT));
    app.run(WindowCreateOptions::new(ui::layout));
}
