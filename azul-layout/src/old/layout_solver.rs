use rayon::prelude::*;
use core::f32;
use alloc::collections::btree_map::BTreeMap;
use alloc::collections::btree_set::BTreeSet;
use alloc::vec::Vec;
use alloc::string::ToString;
use azul_css::*;
use azul_core::{
    traits::GetTextLayout,
    id_tree::{NodeId, NodeDataContainer, NodeDataContainerRef, NodeDataContainerRefMut},
    dom::{NodeData, NodeType},
    styled_dom::{
        StyledDom, DomId, StyledNode, AzNode, StyledNodeState,
        ParentWithNodeDepth, ChangedCssProperty, CssPropertyCache,
    },
    ui_solver::{
        DEFAULT_FONT_SIZE_PX, ScrolledNodes, ResolvedOffsets, LayoutResult,
        PositionedRectangle, OverflowInfo, WhConstraint, WidthCalculatedRect, HeightCalculatedRect,
        HorizontalSolvedPosition, VerticalSolvedPosition,
    },
    app_resources::{
        ResourceUpdate, IdNamespace,
        AppResources, FontInstanceKey,
        ShapedWords, WordPositions, Words,
    },
    callbacks::PipelineId,
    display_list::RenderCallbacks,
    window::{FullWindowState, LogicalRect, LogicalSize, LogicalPosition},
};
use rust_fontconfig::FcFontCache;

const DEFAULT_FLEX_GROW_FACTOR: f32 = 1.0;

#[derive(Debug)]
struct WhConfig {
    width: WidthConfig,
    height: HeightConfig,
}

#[derive(Debug, Default)]
struct WidthConfig {
    exact: Option<LayoutWidth>,
    max: Option<LayoutMaxWidth>,
    min: Option<LayoutMinWidth>,
}

#[derive(Debug, Default)]
struct HeightConfig {
    exact: Option<LayoutHeight>,
    max: Option<LayoutMaxHeight>,
    min: Option<LayoutMinHeight>,
}

fn precalculate_wh_config(styled_dom: &StyledDom) -> NodeDataContainer<WhConfig> {

    use rayon::prelude::*;

    let css_property_cache = styled_dom.get_css_property_cache();
    let node_data_container = styled_dom.node_data.as_container();

    NodeDataContainer {
        internal: styled_dom.styled_nodes
        .as_container().internal
        .par_iter()
        .enumerate()
        .map(|(node_id, styled_node)| {
            let node_id = NodeId::new(node_id);
            WhConfig {
                width: WidthConfig {
                    exact: css_property_cache.get_width(&node_data_container[node_id], &node_id, &styled_node.state).and_then(|p| p.get_property().copied()),
                    max: css_property_cache.get_max_width(&node_data_container[node_id], &node_id, &styled_node.state).and_then(|p| p.get_property().copied()),
                    min: css_property_cache.get_min_width(&node_data_container[node_id], &node_id, &styled_node.state).and_then(|p| p.get_property().copied()),
                },
                height: HeightConfig {
                    exact: css_property_cache.get_height(&node_data_container[node_id], &node_id, &styled_node.state).and_then(|p| p.get_property().copied()),
                    max: css_property_cache.get_max_height(&node_data_container[node_id], &node_id, &styled_node.state).and_then(|p| p.get_property().copied()),
                    min: css_property_cache.get_min_height(&node_data_container[node_id], &node_id, &styled_node.state).and_then(|p| p.get_property().copied()),
                },
            }
        })
        .collect(),
    }
}

macro_rules! determine_preferred {
    ($fn_name:ident, $width:ident) => (

    /// - `preferred_inner_width` denotes the preferred width of the width or height got from the
    /// from the rectangles content.
    ///
    /// For example, if you have an image, the `preferred_inner_width` is the images width,
    /// if the node type is an text, the `preferred_inner_width` is the text height.
    fn $fn_name(config: &WhConfig, preferred_inner_width: Option<f32>, parent_width: f32) -> WhConstraint {

        let mut width = config.$width.exact.as_ref().map(|x| { x.inner.to_pixels(parent_width) });
        let min_width = config.$width.min.as_ref().map(|x| x.inner.to_pixels(parent_width));
        let max_width = config.$width.max.as_ref().map(|x| x.inner.to_pixels(parent_width));

        // TODO: correct for width / height less than 0 - "negative" width is impossible!

        let (absolute_min, absolute_max) = {
            if let (Some(min), Some(max)) = (min_width, max_width) {
                if min_width < max_width {
                    (Some(min), Some(max))
                } else {
                    // min-width > max_width: max_width wins
                    (Some(max), Some(max))
                }
            } else {
                (min_width, max_width)
            }
        };

        // We only need to correct the width if the preferred width is in the range
        // between min & max and the width isn't already specified as a style
        if let Some(preferred_width) = preferred_inner_width {
            if width.is_none() &&
               preferred_width > absolute_min.unwrap_or(0.0) &&
               preferred_width < absolute_max.unwrap_or(f32::MAX)
            {
                width = Some(preferred_width);
            }
        };

        if let Some(width) = width {
            if let Some(max_width) = absolute_max {
                if let Some(min_width) = absolute_min {
                    if min_width < width && width < max_width {
                        // normal: min_width < width < max_width
                        WhConstraint::EqualTo(width)
                    } else if width > max_width {
                        WhConstraint::EqualTo(max_width)
                    } else if width < min_width {
                        WhConstraint::EqualTo(min_width)
                    } else {
                        WhConstraint::Unconstrained /* unreachable */
                    }
                } else {
                    // width & max_width
                    WhConstraint::EqualTo(width.min(max_width))
                }
            } else if let Some(min_width) = absolute_min {
                // no max width, only width & min_width
                WhConstraint::EqualTo(width.max(min_width))
            } else {
                // no min-width or max-width
                WhConstraint::EqualTo(width)
            }
        } else {
            // no width, only min_width and max_width
            if let Some(max_width) = absolute_max {
                if let Some(min_width) = absolute_min {
                    WhConstraint::Between(min_width, max_width)
                } else {
                    // TODO: check sign positive on max_width!
                    WhConstraint::Between(0.0, max_width)
                }
            } else {
                if let Some(min_width) = absolute_min {
                    WhConstraint::Between(min_width, f32::MAX)
                } else {
                    // no width, min_width or max_width
                    WhConstraint::Unconstrained
                }
            }
        }
    })
}

// Returns the preferred width, given [width, min_width, max_width] inside a RectLayout
// or `None` if the height can't be determined from the node alone.
//
// fn determine_preferred_width(layout: &RectLayout) -> Option<f32>
determine_preferred!(determine_preferred_width, width);

// Returns the preferred height, given [height, min_height, max_height] inside a RectLayout
// or `None` if the height can't be determined from the node alone.
//
// fn determine_preferred_height(layout: &RectLayout) -> Option<f32>
determine_preferred!(determine_preferred_height, height);

/// ```rust
/// typed_arena!(
///     WidthCalculatedRect,
///     preferred_width,
///     determine_preferred_width,
///     get_horizontal_padding,
///     get_flex_basis_horizontal,
///     width_calculated_rect_arena_from_rect_layout_arena,
///     bubble_preferred_widths_to_parents,
///     width_calculated_rect_arena_apply_flex_grow,
///     width_calculated_rect_arena_sum_children_flex_basis,
///     Horizontal,
/// )
/// ```
macro_rules! typed_arena {(
    $struct_name:ident,
    $preferred_field:ident,
    $determine_preferred_fn:ident,
    $get_padding_fn:ident,
    $get_flex_basis:ident,
    $from_rect_layout_arena_fn_name:ident,
    $bubble_fn_name:ident,
    $apply_flex_grow_fn_name:ident,
    $sum_children_flex_basis_fn_name:ident,
    $main_axis:ident,
    $margin_left:ident,
    $margin_right:ident,
    $padding_left:ident,
    $padding_right:ident,
    $left:ident,
    $right:ident,
) => (


    /// Fill out the preferred width of all nodes.
    ///
    /// We could operate on the NodeDataContainer<StyledNode> directly, but that makes testing very
    /// hard since we are only interested in testing or touching the layout. So this makes the
    /// calculation maybe a few microseconds slower, but gives better testing capabilities
    ///
    /// NOTE: Later on, this could maybe be a NodeDataContainer<&'a RectLayout>.
    #[must_use]
    fn $from_rect_layout_arena_fn_name<'a>(
        wh_configs: &NodeDataContainerRef<'a, WhConfig>,
        offsets: &NodeDataContainerRef<'a, AllOffsets>,
        widths: &NodeDataContainerRef<'a, Option<f32>>,
        node_hierarchy: &NodeDataContainerRef<'a, AzNode>,
        node_depths: &[ParentWithNodeDepth],
        root_size_width: f32,
    ) -> NodeDataContainer<$struct_name> {

        // then calculate the widths again, but this time using the parent nodes
        let mut new_nodes = NodeDataContainer {
            internal: vec![$struct_name::default();node_hierarchy.len()]
        };

        for ParentWithNodeDepth { depth: _, node_id } in node_depths.iter() {

            let parent_id = match node_id.into_crate_internal() { Some(s) => s, None => continue, };

            let nd = &wh_configs[parent_id];
            let parent_offsets = &offsets[parent_id];
            let width = match widths.get(parent_id) { Some(s) => *s, None => continue, };

            let parent_width = node_hierarchy
            .get(parent_id)
            .and_then(|t| new_nodes.as_ref().get(t.parent_id()?).map(|parent| parent.$preferred_field))
            .unwrap_or_default()
            .max_available_space()
            .unwrap_or(root_size_width);

            let parent_width = $determine_preferred_fn(&nd, width, parent_width);

            new_nodes.as_ref_mut()[parent_id] = $struct_name {
                // TODO: get the initial width of the rect content
                $preferred_field: parent_width,

                $margin_left: parent_offsets.margin.$left.as_ref().copied(),
                $margin_right: parent_offsets.margin.$right.as_ref().copied(),
                $padding_left: parent_offsets.padding.$left.as_ref().copied(),
                $padding_right: parent_offsets.padding.$right.as_ref().copied(),
                $left: parent_offsets.position.$left.as_ref().copied(),
                $right: parent_offsets.position.$right.as_ref().copied(),

                flex_grow_px: 0.0,
                min_inner_size_px: 0.0,
            };

            for child_id in parent_id.az_children(node_hierarchy) {
                let nd = &wh_configs[child_id];
                let child_offsets = &offsets[child_id];
                let width = match widths.get(child_id) { Some(s) => *s, None => continue, };
                new_nodes.as_ref_mut()[child_id] = $struct_name {
                    // TODO: get the initial width of the rect content
                    $preferred_field: $determine_preferred_fn(&nd, width, parent_width.max_available_space().unwrap_or(0.0)),

                    $margin_left: child_offsets.margin.$left.as_ref().copied(),
                    $margin_right: child_offsets.margin.$right.as_ref().copied(),
                    $padding_left: child_offsets.padding.$left.as_ref().copied(),
                    $padding_right: child_offsets.padding.$right.as_ref().copied(),
                    $left: child_offsets.position.$left.as_ref().copied(),
                    $right: child_offsets.position.$right.as_ref().copied(),

                    flex_grow_px: 0.0,
                    min_inner_size_px: 0.0,
                }
            }
        }

        new_nodes
    }

    /// Bubble the inner sizes to their parents -  on any parent nodes, fill out
    /// the width so that the `preferred_width` can contain the child nodes (if
    /// that doesn't violate the constraints of the parent)
    fn $bubble_fn_name<'a, 'b>(
        node_data: &mut NodeDataContainerRefMut<'b, $struct_name>,
        node_hierarchy: &NodeDataContainerRef<'a, AzNode>,
        layout_positions: &NodeDataContainerRef<'a, LayoutPosition>,
        node_depths: &[ParentWithNodeDepth],
    ) {
        // Reverse, since we want to go from the inside out (depth 5 needs to be filled out first)
        //
        // Set the preferred_width of the parent nodes
        for ParentWithNodeDepth { depth: _, node_id } in node_depths.iter().rev() {

            use self::WhConstraint::*;

            let non_leaf_id = match node_id.into_crate_internal() { Some(s) => s, None => continue, };

            let parent_width = node_hierarchy[non_leaf_id].parent_id().and_then(|parent_id| node_data[parent_id].$preferred_field.min_needed_space()).unwrap_or(0.0);

            // Sum of the direct children's flex-basis = the parents preferred width
            let children_flex_basis = $sum_children_flex_basis_fn_name(node_data, non_leaf_id, node_hierarchy, layout_positions);

            // Calculate the new flex-basis width
            let parent_width_metrics = node_data[non_leaf_id];

            // For calculating the inner width, subtract the parents padding
            let parent_padding = node_data[non_leaf_id].$get_padding_fn(parent_width);

            // If the children are larger than the parents preferred max-width or smaller
            // than the parents min-width, adjust
            let child_width = match parent_width_metrics.$preferred_field {
                Between(min, max) => {
                    if children_flex_basis > (max - parent_padding)  {
                        max
                    } else if children_flex_basis < (min + parent_padding) {
                        min
                    } else {
                        children_flex_basis
                    }
                },
                EqualTo(exact) => exact - parent_padding,
                Unconstrained => children_flex_basis,
            };

            node_data[non_leaf_id].min_inner_size_px = child_width;
        }

        // Now, the width of all elements should be filled,
        // but they aren't flex-growed or flex-shrinked yet
    }

    /// Go from the root down and flex_grow the children if needed - respects the `width`, `min_width` and `max_width` properties
    /// The layout step doesn't account for the min_width and max_width constraints, so we have to adjust them manually
    fn $apply_flex_grow_fn_name<'a, 'b>(
        node_data: &mut NodeDataContainerRefMut<'b, $struct_name>,
        node_hierarchy: &NodeDataContainerRef<'a, AzNode>,
        layout_flex_grows: &NodeDataContainerRef<'a, f32>,
        layout_positions: &NodeDataContainerRef<'a, LayoutPosition>,
        layout_directions: &NodeDataContainerRef<'a, LayoutFlexDirection>,
        node_depths: &[ParentWithNodeDepth],
        root_width: f32,
        parents_to_recalc: &BTreeSet<NodeId>,
    ) {
        /// Does the actual width layout, respects the `width`, `min_width` and `max_width`
        /// properties as well as the `flex_grow` factor. `flex_shrink` currently does nothing.
        fn distribute_space_along_main_axis<'a>(
            node_id: &NodeId,
            node_hierarchy: &NodeDataContainerRef<'a, AzNode>,
            layout_flex_grows: &NodeDataContainerRef<'a, f32>,
            layout_positions: &NodeDataContainerRef<'a, LayoutPosition>,
            width_calculated_arena: &'a mut NodeDataContainerRefMut<$struct_name>,
            positioned_node_stack: &[NodeId],
            root_width: f32
        ) {
            // The inner space of the parent node, without the padding
            let mut parent_node_inner_width = {
                let parent_node = &width_calculated_arena[*node_id];
                let parent_parent_width = node_hierarchy[*node_id].parent_id().and_then(|p| width_calculated_arena[p].$preferred_field.max_available_space()).unwrap_or(root_width);
                parent_node.min_inner_size_px + parent_node.flex_grow_px - parent_node.$get_padding_fn(parent_parent_width)
            };

            // 1. Set all child elements that have an exact width to that width, record their violations
            //    and add their violation to the leftover horizontal space.
            // let mut horizontal_space_from_fixed_width_items = 0.0;
            let mut horizontal_space_taken_up_by_fixed_width_items = 0.0;

            {
                // Vec<(NodeId, PreferredWidth)>
                let exact_width_childs = node_id
                        .az_children(node_hierarchy)
                        .filter_map(|id| if let WhConstraint::EqualTo(exact) = width_calculated_arena[id].$preferred_field {
                            Some((id, exact))
                        } else {
                            None
                        })
                        .collect::<Vec<(NodeId, f32)>>();

                for (exact_width_child_id, exact_width) in exact_width_childs {

                    // If this child node is `position: absolute`, it doesn't take any space away from
                    // its siblings, since it is taken out of the regular content flow
                    if layout_positions[exact_width_child_id] != LayoutPosition::Absolute {
                        horizontal_space_taken_up_by_fixed_width_items += exact_width;
                    }

                    // so that node.min_inner_size_px + node.flex_grow_px = exact_width
                    let flex_grow_px = exact_width - width_calculated_arena[exact_width_child_id].min_inner_size_px;
                    width_calculated_arena[exact_width_child_id].flex_grow_px = flex_grow_px;
                }
            }

            // The fixed-width items are now considered solved, so subtract them out of the width of the parent.
            parent_node_inner_width -= horizontal_space_taken_up_by_fixed_width_items;

            // Now we can be sure that if we write #x { width: 500px; } that it will actually be 500px large
            // and not be influenced by flex in any way.

            // 2. Set all items to their minimum width. Record how much space is gained by doing so.
            let mut horizontal_space_taken_up_by_variable_items = 0.0;

            let mut variable_width_childs = node_id
                .az_children(node_hierarchy)
                .filter(|id| !width_calculated_arena[*id].$preferred_field.is_fixed_constraint())
                .collect::<BTreeSet<NodeId>>();

            let mut absolute_variable_width_nodes = Vec::new();

            for variable_child_id in &variable_width_childs {

                if layout_positions[*variable_child_id] != LayoutPosition::Absolute {

                    let min_width = width_calculated_arena[*variable_child_id].$preferred_field.min_needed_space().unwrap_or(0.0);

                    horizontal_space_taken_up_by_variable_items += min_width;

                    // so that node.min_inner_size_px + node.flex_grow_px = min_width
                    let flex_grow_px = min_width - width_calculated_arena[*variable_child_id].min_inner_size_px;
                    width_calculated_arena[*variable_child_id].flex_grow_px = flex_grow_px;

                } else {

                    // `position: absolute` items don't take space away from their siblings, rather
                    // they take the minimum needed space by their content

                    let root_id = NodeId::new(0);
                    let nearest_relative_parent_node = positioned_node_stack.get(positioned_node_stack.len() - 1).unwrap_or(&root_id);
                    let relative_parent_width = {
                        let relative_parent_node = &width_calculated_arena[*nearest_relative_parent_node];
                        relative_parent_node.flex_grow_px + relative_parent_node.min_inner_size_px
                    };

                    // By default, absolute positioned elements take the width of their content
                    // let min_inner_width = width_calculated_arena[*variable_child_id].$preferred_field.min_needed_space().unwrap_or(0.0);

                    // The absolute positioned node might have a max-width constraint, which has a
                    // higher precedence than `top, bottom, left, right`.
                    let max_space_current_node = match width_calculated_arena[*variable_child_id].$preferred_field {
                        WhConstraint::EqualTo(e) => e,
                        WhConstraint::Between(min, max) => {
                            if relative_parent_width > min {
                                if relative_parent_width < max {
                                    relative_parent_width
                                } else {
                                    max
                                }
                            } else {
                                min
                            }
                        },
                        WhConstraint::Unconstrained => relative_parent_width,
                    };

                    // so that node.min_inner_size_px + node.flex_grow_px = max_space_current_node
                    let flex_grow_px = max_space_current_node - width_calculated_arena[*variable_child_id].min_inner_size_px;
                    width_calculated_arena[*variable_child_id].flex_grow_px = flex_grow_px;

                    absolute_variable_width_nodes.push(*variable_child_id);
                }

            }

            // Absolute positioned nodes aren't in the space-to-distribute set
            for absolute_node in absolute_variable_width_nodes {
                variable_width_childs.remove(&absolute_node);
            }

            // This satisfies the `width` and `min_width` constraints. However, we still need to worry about
            // the `max_width` and unconstrained children.
            //
            // By setting the items to their minimum size, we've gained some space that we now need to distribute
            // according to the flex_grow values
            parent_node_inner_width -= horizontal_space_taken_up_by_variable_items;

            let mut total_horizontal_space_available = parent_node_inner_width;
            let mut max_width_violations = Vec::new();

            loop {

                if total_horizontal_space_available <= 0.0 || variable_width_childs.is_empty() {
                    break;
                }

                // In order to apply flex-grow correctly, we need the sum of
                // the flex-grow factors of all the variable-width children
                //
                // NOTE: variable_width_childs can change its length, have to recalculate every loop!
                let children_combined_flex_grow: f32 = variable_width_childs
                    .iter()
                    .map(|child_id| layout_flex_grows[*child_id])
                    .sum();

                let size_per_child = total_horizontal_space_available as f32 / children_combined_flex_grow;

                // Grow all variable children by the same amount.
                for variable_child_id in &variable_width_childs {

                    let flex_grow_of_child = layout_flex_grows[*variable_child_id];

                    let added_space_for_one_child = size_per_child * flex_grow_of_child;

                    let current_width_of_child = width_calculated_arena[*variable_child_id].total();

                    let mut has_max_width_constraint = None;

                    if let Some(max_width) = width_calculated_arena[*variable_child_id].$preferred_field.max_available_space() {
                        if (current_width_of_child + added_space_for_one_child) > max_width {
                            has_max_width_constraint = Some(max_width);
                        }
                    }

                    if let Some(max_width) = has_max_width_constraint {
                        // grow so that node.min_inner_size_px + node.flex_grow_px = max_width
                        let flex_grow_px = max_width - width_calculated_arena[*variable_child_id].min_inner_size_px;
                        width_calculated_arena[*variable_child_id].flex_grow_px = flex_grow_px;

                        max_width_violations.push(*variable_child_id);
                    } else {
                        // grow so that node.min_inner_size_px + node.flex_grow_px = added_space_for_one_child
                        let flex_grow_px = added_space_for_one_child - width_calculated_arena[*variable_child_id].min_inner_size_px;
                        width_calculated_arena[*variable_child_id].flex_grow_px = flex_grow_px;
                    }
                }

                // If we haven't violated any max_width constraints, then we have
                // added all remaining widths and thereby successfully solved the layout
                if max_width_violations.is_empty() {
                    break;
                } else {
                    // Nodes that were violated can't grow anymore in the next iteration,
                    // so we remove them from the solution and consider them "solved".
                    // Their amount of violation then gets distributed across the remaining
                    // items in the next iteration.
                    for solved_node_id in max_width_violations.drain(..) {

                        // Since the node now gets removed, it doesn't contribute to the pool anymore
                        total_horizontal_space_available -= width_calculated_arena[solved_node_id].total();

                        variable_width_childs.remove(&solved_node_id);
                    }
                }
            }
        }

        fn distribute_space_along_cross_axis<'a>(
            node_id: &NodeId,
            node_hierarchy: &NodeDataContainerRef<'a, AzNode>,
            layout_positions: &NodeDataContainerRef<'a, LayoutPosition>,
            width_calculated_arena: &'a mut NodeDataContainerRefMut< $struct_name>,
            positioned_node_stack: &[NodeId],
            root_width: f32
        ) {
            // The inner space of the parent node, without the padding
            let parent_node_inner_width = {
                let parent_node = &width_calculated_arena[*node_id];
                let parent_parent_width = node_hierarchy[*node_id].parent_id().and_then(|p| width_calculated_arena[p].$preferred_field.max_available_space()).unwrap_or(root_width);
                parent_node.min_inner_size_px + parent_node.flex_grow_px - parent_node.$get_padding_fn(parent_parent_width)
            };

            let last_relative_node_width = {
                let zero_node = NodeId::new(0);
                let last_relative_node_id = positioned_node_stack.get(positioned_node_stack.len() - 1).unwrap_or(&zero_node);
                let last_relative_node = &width_calculated_arena[*last_relative_node_id];
                let last_relative_node_parent_width = node_hierarchy[*last_relative_node_id].parent_id().and_then(|p| width_calculated_arena[p].$preferred_field.max_available_space()).unwrap_or(root_width);
                last_relative_node.min_inner_size_px + last_relative_node.flex_grow_px - last_relative_node.$get_padding_fn(last_relative_node_parent_width)
            };

            for child_id in node_id.az_children(node_hierarchy) {

                let parent_node_inner_width = if layout_positions[child_id] != LayoutPosition::Absolute {
                    parent_node_inner_width
                } else {
                    last_relative_node_width
                };

                let preferred_width = {
                    let min_width = width_calculated_arena[child_id].$preferred_field.min_needed_space().unwrap_or(0.0);
                    // In this case we want to overflow if the min width of the cross axis
                    if min_width > parent_node_inner_width {
                        min_width
                    } else {
                        if let Some(max_width) = width_calculated_arena[child_id].$preferred_field.max_available_space() {
                            if max_width > parent_node_inner_width {
                                parent_node_inner_width
                            } else {
                                max_width
                            }
                        } else {
                            parent_node_inner_width
                        }
                    }
                };

                // so that node.min_inner_size_px + node.flex_grow_px = preferred_width
                let flex_grow_px = preferred_width - width_calculated_arena[child_id].min_inner_size_px;
                width_calculated_arena[child_id].flex_grow_px = flex_grow_px;
            }
        }

        debug_assert!(node_data[NodeId::ZERO].flex_grow_px == 0.0);

        // Set the window width on the root node (since there is only one root node, we can
        // calculate the `flex_grow_px` directly)
        //
        // Usually `top_level_flex_basis` is NOT 0.0, rather it's the sum of all widths in the DOM,
        // i.e. the sum of the whole DOM tree
        let top_level_flex_basis = node_data[NodeId::ZERO].min_inner_size_px;

        // The root node can still have some sort of max-width attached, so we need to check for that
        let root_preferred_width = if let Some(max_width) = node_data[NodeId::ZERO].$preferred_field.max_available_space() {
            if root_width > max_width { max_width } else { root_width }
        } else {
            root_width
        };

        node_data[NodeId::ZERO].flex_grow_px = root_preferred_width - top_level_flex_basis;

        // Keep track of the nearest relative or absolute positioned element
        let mut positioned_node_stack = vec![NodeId::new(0)];

        for ParentWithNodeDepth { depth: _, node_id } in node_depths.iter() {

            use azul_css::{LayoutAxis, LayoutPosition};

            let parent_id = match node_id.into_crate_internal() { Some(s) => s, None => continue, };

            if !parents_to_recalc.contains(&parent_id) { continue; }

            let parent_is_positioned = layout_positions[parent_id] != LayoutPosition::Static;
            if parent_is_positioned {
                positioned_node_stack.push(parent_id);
            }

            if layout_directions[parent_id].get_axis() == LayoutAxis::$main_axis {
                distribute_space_along_main_axis(&parent_id, node_hierarchy, layout_flex_grows, layout_positions, node_data, &positioned_node_stack, root_width);
            } else {
                distribute_space_along_cross_axis(&parent_id, node_hierarchy, layout_positions, node_data, &positioned_node_stack, root_width);
            }

            if parent_is_positioned {
                positioned_node_stack.pop();
            }
        }
    }

    /// Returns the sum of the flex-basis of the current nodes' children
    #[must_use]
    fn $sum_children_flex_basis_fn_name<'a, 'b>(
        node_data: &mut NodeDataContainerRefMut<'b, $struct_name>,
        node_id: NodeId,
        node_hierarchy: &NodeDataContainerRef<'a, AzNode>,
        layout_positions: &NodeDataContainerRef<'a, LayoutPosition>,
    ) -> f32 {
        let parent_width = node_data[node_id].$preferred_field.max_available_space().unwrap_or(0.0);
        node_id
            .az_children(node_hierarchy)
            .filter(|child_node_id| layout_positions[*child_node_id] != LayoutPosition::Absolute)
            .map(|child_node_id| node_data[child_node_id].$get_flex_basis(parent_width))
            .sum()
    }

)}

typed_arena!(
    WidthCalculatedRect,
    preferred_width,
    determine_preferred_width,
    get_horizontal_padding,
    get_flex_basis_horizontal,
    width_calculated_rect_arena_from_rect_layout_arena,
    bubble_preferred_widths_to_parents,
    width_calculated_rect_arena_apply_flex_grow,
    width_calculated_rect_arena_sum_children_flex_basis,
    Horizontal,
    margin_left,
    margin_right,
    padding_left,
    padding_right,
    left,
    right,
);

typed_arena!(
    HeightCalculatedRect,
    preferred_height,
    determine_preferred_height,
    get_vertical_padding,
    get_flex_basis_vertical,
    height_calculated_rect_arena_from_rect_layout_arena,
    bubble_preferred_heights_to_parents,
    height_calculated_rect_arena_apply_flex_grow,
    height_calculated_rect_arena_sum_children_flex_basis,
    Vertical,
    margin_top,
    margin_bottom,
    padding_top,
    padding_bottom,
    top,
    bottom,
);

/// Returns the solved widths of the items in a BTree form
pub(crate) fn solve_flex_layout_width<'a, 'b>(
    width_calculated_arena: &'a mut NodeDataContainerRefMut<'a, WidthCalculatedRect>,
    layout_flex_grow: &NodeDataContainerRef<'a, f32>,
    layout_positions: &NodeDataContainerRef<'a, LayoutPosition>,
    layout_directions: &NodeDataContainerRef<'a, LayoutFlexDirection>,
    node_hierarchy: &'b NodeDataContainerRef<'a, AzNode>,
    node_depths: &[ParentWithNodeDepth],
    window_width: f32,
    parents_to_recalc: &BTreeSet<NodeId>,
) {
    bubble_preferred_widths_to_parents(width_calculated_arena, node_hierarchy, layout_positions, node_depths);
    width_calculated_rect_arena_apply_flex_grow(width_calculated_arena, node_hierarchy, layout_flex_grow, layout_positions, layout_directions, node_depths, window_width, parents_to_recalc);
}

/// Returns the solved height of the items in a BTree form
pub(crate) fn solve_flex_layout_height<'a, 'b>(
    height_calculated_arena: &'a mut NodeDataContainerRefMut<'a, HeightCalculatedRect>,
    layout_flex_grow: &NodeDataContainerRef<'a, f32>,
    layout_positions: &NodeDataContainerRef<'a, LayoutPosition>,
    layout_directions: &NodeDataContainerRef<'a, LayoutFlexDirection>,
    node_hierarchy: &'b NodeDataContainerRef<'a, AzNode>,
    node_depths: &[ParentWithNodeDepth],
    window_height: f32,
    parents_to_recalc: &BTreeSet<NodeId>,
) {
    bubble_preferred_heights_to_parents(height_calculated_arena, node_hierarchy, layout_positions, node_depths);
    height_calculated_rect_arena_apply_flex_grow(height_calculated_arena, node_hierarchy, layout_flex_grow, layout_positions, layout_directions, node_depths, window_height, parents_to_recalc);
}

macro_rules! get_position {
($fn_name:ident,
 $width_layout:ident,
 $height_solved_position:ident,
 $solved_widths_field:ident,
 $left:ident,
 $right:ident,
 $margin_left:ident,
 $margin_right:ident,
 $padding_left:ident,
 $padding_right:ident,
 $axis:ident
) => (

/// Traverses along the DOM and solve for the X or Y position
fn $fn_name<'a>(
    arena: &mut NodeDataContainerRefMut<'a, $height_solved_position>,
    node_hierarchy: &NodeDataContainerRef<'a, AzNode>,
    layout_positions: &NodeDataContainerRef<'a, LayoutPosition>,
    layout_directions: &NodeDataContainerRef<'a, LayoutFlexDirection>,
    layout_justify_contents: &NodeDataContainerRef<'a, LayoutJustifyContent>,
    node_depths: &[ParentWithNodeDepth],
    solved_widths: &NodeDataContainerRef<'a, $width_layout>,
    parents_to_solve: &BTreeSet<NodeId>
) {

    fn determine_child_x_absolute<'a>(
        child_id: NodeId,
        positioned_node_stack: &[NodeId],
        arena_solved_data: &mut NodeDataContainerRefMut<'a, $height_solved_position>,
        solved_widths: &NodeDataContainerRef<'a, $width_layout>,
        node_hierarchy: &NodeDataContainerRef<'a, AzNode>,
    ) {
        let child_width_with_padding = {
            let child_node = &solved_widths[child_id];
            child_node.min_inner_size_px + child_node.flex_grow_px
        };

        let child_node = &solved_widths[child_id];
        let child_node_parent_width = node_hierarchy[child_id].parent_id().map(|p| solved_widths[p].total()).unwrap_or(0.0) as f32;
        let child_margin_left = child_node.$margin_left.and_then(|x| Some(x.get_property()?.inner.to_pixels(child_node_parent_width))).unwrap_or(0.0);
        let child_margin_right = child_node.$margin_right.and_then(|x| Some(x.get_property()?.inner.to_pixels(child_node_parent_width))).unwrap_or(0.0);

        let zero_node = NodeId::new(0);
        let last_relative_node_id = positioned_node_stack.get(positioned_node_stack.len() - 1).unwrap_or(&zero_node);

        let last_relative_node = &solved_widths[*last_relative_node_id];
        let last_relative_padding_left = last_relative_node.$padding_left.and_then(|x| Some(x.get_property()?.inner.to_pixels(child_node_parent_width))).unwrap_or(0.0);
        let last_relative_padding_right = last_relative_node.$padding_right.and_then(|x| Some(x.get_property()?.inner.to_pixels(child_node_parent_width))).unwrap_or(0.0);

        let last_relative_node_x = arena_solved_data[*last_relative_node_id].0 + last_relative_padding_left;
        let last_relative_node_inner_width = {
            let last_relative_node = &solved_widths[*last_relative_node_id];
            last_relative_node.min_inner_size_px + last_relative_node.flex_grow_px - (last_relative_padding_left + last_relative_padding_right)
        };

        let child_left = child_node.$left.and_then(|s| Some(s.get_property()?.inner.to_pixels(child_node_parent_width)));
        let child_right = child_node.$right.and_then(|s| Some(s.get_property()?.inner.to_pixels(child_node_parent_width)));

        if let Some(child_right) = child_right {
            // align right / bottom of last relative parent
            arena_solved_data[child_id].0 =
                last_relative_node_x
                + last_relative_node_inner_width
                - child_width_with_padding
                - child_margin_right
                - child_right;
        } else {
            // align left / top of last relative parent
            arena_solved_data[child_id].0 =
                last_relative_node_x
                + child_margin_left
                + child_left.unwrap_or(0.0);
        }
    }

    fn determine_child_x_along_main_axis<'a>(
        main_axis_alignment: LayoutJustifyContent,
        layout_positions: &NodeDataContainerRef<'a, LayoutPosition>,
        arena_solved_data: &mut NodeDataContainerRefMut<'a, $height_solved_position>,
        solved_widths: &NodeDataContainerRef<'a, $width_layout>,
        child_id: NodeId,
        parent_x_position: f32,
        parent_inner_width: f32,
        sum_x_of_children_so_far: &mut f32,
        positioned_node_stack: &[NodeId],
        node_hierarchy: &NodeDataContainerRef<'a, AzNode>,
    ) {
        use azul_css::LayoutJustifyContent::*;

        let child_width_with_padding = {
            let child_node = &solved_widths[child_id];
            child_node.min_inner_size_px + child_node.flex_grow_px
        };

        // width: increase X according to the main axis, Y according to the cross_axis
        let child_node = &solved_widths[child_id];
        let child_node_parent_width = node_hierarchy[child_id].parent_id().map(|p| solved_widths[p].total()).unwrap_or(0.0) as f32;
        let child_margin_left = child_node.$margin_left.and_then(|x| Some(x.get_property()?.inner.to_pixels(child_node_parent_width))).unwrap_or(0.0);
        let child_margin_right = child_node.$margin_right.and_then(|x| Some(x.get_property()?.inner.to_pixels(child_node_parent_width))).unwrap_or(0.0);

        if layout_positions[child_id] == LayoutPosition::Absolute {
            determine_child_x_absolute(
                child_id,
                positioned_node_stack,
                arena_solved_data,
                solved_widths,
                node_hierarchy,
            );
        } else {
            // X position of the top left corner
            // WARNING: End has to be added after all children!
            let x_of_top_left_corner = match main_axis_alignment {
                Start | End => {
                    parent_x_position + *sum_x_of_children_so_far + child_margin_left
                },
                Center => {
                    parent_x_position
                    + ((parent_inner_width as f32 / 2.0)
                    - ((*sum_x_of_children_so_far + child_margin_right + child_width_with_padding) as f32 / 2.0))
                },
                SpaceBetween => {
                    parent_x_position // TODO!
                },
                SpaceAround => {
                    parent_x_position // TODO!
                },
                SpaceEvenly => {
                    parent_x_position // TODO!
                },
            };

            arena_solved_data[child_id].0 = x_of_top_left_corner;
            *sum_x_of_children_so_far += child_margin_right + child_width_with_padding + child_margin_left;
        }
    }

    fn determine_child_x_along_cross_axis<'a>(
        layout_positions: &NodeDataContainerRef<'a, LayoutPosition>,
        solved_widths: &NodeDataContainerRef<'a, $width_layout>,
        child_id: NodeId,
        positioned_node_stack: &[NodeId],
        arena_solved_data: &mut NodeDataContainerRefMut<'a, $height_solved_position>,
        parent_x_position: f32,
        node_hierarchy: &NodeDataContainerRef<'a, AzNode>
    ) {
        let child_node = &solved_widths[child_id];
        let child_node_parent_width = node_hierarchy[child_id].parent_id().map(|p| solved_widths[p].total()).unwrap_or(0.0) as f32;
        let child_margin_left = child_node.$margin_left.and_then(|x| Some(x.get_property()?.inner.to_pixels(child_node_parent_width))).unwrap_or(0.0);

        if layout_positions[child_id] == LayoutPosition::Absolute {
            determine_child_x_absolute(
                child_id,
                positioned_node_stack,
                arena_solved_data,
                solved_widths,
                node_hierarchy,
            );
        } else {
            arena_solved_data[child_id].0 = parent_x_position + child_margin_left;
        }
    }

    use azul_css::{LayoutAxis, LayoutJustifyContent};

    // Stack of the positioned nodes (nearest relative or absolute positioned node)
    let mut positioned_node_stack = vec![NodeId::new(0)];

    for ParentWithNodeDepth { depth: _, node_id } in node_depths.iter() {

        let parent_id = match node_id.into_crate_internal() { Some(s) => s, None => continue, };
        if !parents_to_solve.contains(&parent_id) { continue; }
        let parent_node = &solved_widths[parent_id];
        let parent_parent_width = node_hierarchy[parent_id].parent_id().map(|p| solved_widths[p].total()).unwrap_or(0.0) as f32;
        let parent_padding_left = parent_node.$padding_left.and_then(|x| Some(x.get_property()?.inner.to_pixels(parent_parent_width))).unwrap_or(0.0) as f32;
        let parent_padding_right = parent_node.$padding_right.and_then(|x| Some(x.get_property()?.inner.to_pixels(parent_parent_width))).unwrap_or(0.0) as f32;
        let parent_x_position = arena[parent_id].0 + parent_padding_left;
        let parent_direction = layout_directions[parent_id];

        // Push nearest relative or absolute positioned element
        let parent_is_positioned = layout_positions[parent_id] != LayoutPosition::Static;
        if parent_is_positioned {
            positioned_node_stack.push(parent_id);
        }

        let parent_inner_width = {
            parent_node.total() - (parent_padding_left + parent_padding_right)
        };

        if parent_direction.get_axis() == LayoutAxis::$axis {
            // Along main axis: Take X of parent
            let main_axis_alignment = layout_justify_contents[parent_id];
            let mut sum_x_of_children_so_far = 0.0;

            if parent_direction.is_reverse() {
                for child_id in parent_id.az_reverse_children(node_hierarchy) {
                    determine_child_x_along_main_axis(
                        main_axis_alignment,
                        layout_positions,
                        arena,
                        solved_widths,
                        child_id,
                        parent_x_position,
                        parent_inner_width,
                        &mut sum_x_of_children_so_far,
                        &positioned_node_stack,
                        node_hierarchy,
                    );
                }
            } else {
                for child_id in parent_id.az_children(node_hierarchy) {
                    determine_child_x_along_main_axis(
                        main_axis_alignment,
                        layout_positions,
                        arena,
                        solved_widths,
                        child_id,
                        parent_x_position,
                        parent_inner_width,
                        &mut sum_x_of_children_so_far,
                        &positioned_node_stack,
                        node_hierarchy,
                    );
                }
            }

            // If the direction is `flex-end`, we can't add the X position during the iteration,
            // so we have to "add" the diff to the parent_inner_width at the end
            let should_align_towards_end =
                (!parent_direction.is_reverse() && main_axis_alignment == LayoutJustifyContent::End) ||
                (parent_direction.is_reverse() && main_axis_alignment == LayoutJustifyContent::Start);

            if should_align_towards_end {
                let diff = parent_inner_width - sum_x_of_children_so_far;
                for child_id in parent_id.az_children(node_hierarchy).filter(|ch| layout_positions[*ch] != LayoutPosition::Absolute) {
                    arena[child_id].0 += diff;
                }
            }

        } else {
            // Along cross axis: Increase X with width of current element

            if parent_direction.is_reverse() {
                for child_id in parent_id.az_reverse_children(node_hierarchy) {
                    determine_child_x_along_cross_axis(
                        layout_positions,
                        solved_widths,
                        child_id,
                        &positioned_node_stack,
                        arena,
                        parent_x_position,
                        node_hierarchy,
                    );
                }
            } else {
                for child_id in parent_id.az_children(node_hierarchy) {
                    determine_child_x_along_cross_axis(
                        layout_positions,
                        solved_widths,
                        child_id,
                        &positioned_node_stack,
                        arena,
                        parent_x_position,
                        node_hierarchy,
                    );
                }
            }
        }

        if parent_is_positioned {
            positioned_node_stack.pop();
        }

    }
}

)}

fn get_x_positions<'a>(
    arena: &mut NodeDataContainerRefMut<'a, HorizontalSolvedPosition>,
    solved_widths: &NodeDataContainerRef<'a, WidthCalculatedRect>,
    node_hierarchy: &NodeDataContainerRef<'a, AzNode>,
    layout_positions: &NodeDataContainerRef<'a, LayoutPosition>,
    layout_directions: &NodeDataContainerRef<'a, LayoutFlexDirection>,
    layout_justify_contents: &NodeDataContainerRef<'a, LayoutJustifyContent>,
    node_depths: &[ParentWithNodeDepth],
    origin: LogicalPosition,
    parents_to_solve: &BTreeSet<NodeId>,
) {
    get_position!(
        get_pos_x,
        WidthCalculatedRect,
        HorizontalSolvedPosition,
        solved_widths,
        left,
        right,
        margin_left,
        margin_right,
        padding_left,
        padding_right,
        Horizontal
    );

    get_pos_x(arena, node_hierarchy, layout_positions, layout_directions, layout_justify_contents, node_depths, solved_widths, &parents_to_solve);

    // Add the origin on top of the position
    for item in arena.internal.iter_mut() { item.0 += origin.x; }
}

fn get_y_positions<'a>(
    arena: &mut NodeDataContainerRefMut<'a, VerticalSolvedPosition>,
    solved_heights: &NodeDataContainerRef<'a, HeightCalculatedRect>,
    node_hierarchy: &NodeDataContainerRef<'a, AzNode>,
    layout_positions: &NodeDataContainerRef<'a, LayoutPosition>,
    layout_directions: &NodeDataContainerRef<'a, LayoutFlexDirection>,
    layout_justify_contents: &NodeDataContainerRef<'a, LayoutJustifyContent>,
    node_depths: &[ParentWithNodeDepth],
    origin: LogicalPosition,
    parents_to_solve: &BTreeSet<NodeId>,
) {
    get_position!(
        get_pos_y,
        HeightCalculatedRect,
        VerticalSolvedPosition,
        solved_heights,
        top,
        bottom,
        margin_top,
        margin_bottom,
        padding_top,
        padding_bottom,
        Vertical
    );

    get_pos_y(arena, node_hierarchy, layout_positions, layout_directions, layout_justify_contents, node_depths, solved_heights, &parents_to_solve);

    // Add the origin on top of the position
    for item in arena.internal.iter_mut() { item.0 += origin.y; }
}

#[inline]
pub fn get_layout_positions<'a>(styled_dom: &StyledDom) -> NodeDataContainer<LayoutPosition> {
    let cache = styled_dom.get_css_property_cache();
    let node_data_container = styled_dom.node_data.as_container();
    let styled_nodes = styled_dom.styled_nodes.as_container();
    assert!(node_data_container.internal.len() == styled_nodes.internal.len()); // elide bounds checking
    NodeDataContainer {
        internal: styled_nodes.internal.par_iter().enumerate().map(|(node_id, styled_node)| {
            cache.get_position(&node_data_container.internal[node_id], &NodeId::new(node_id), &styled_node.state).unwrap_or_default().get_property_or_default().unwrap_or_default()
        }).collect()
    }
}

#[inline]
pub fn get_layout_justify_contents<'a>(styled_dom: &StyledDom) -> NodeDataContainer<LayoutJustifyContent> {
    let cache = styled_dom.get_css_property_cache();
    let node_data_container = styled_dom.node_data.as_container();
    let styled_nodes = styled_dom.styled_nodes.as_container();
    assert!(node_data_container.internal.len() == styled_nodes.internal.len()); // elide bounds checking

    NodeDataContainer {
        internal: styled_nodes.internal.par_iter().enumerate().map(|(node_id, styled_node)| {
            cache.get_justify_content(&node_data_container.internal[node_id], &NodeId::new(node_id), &styled_node.state).unwrap_or_default().get_property_or_default().unwrap_or_default()
        }).collect()
    }
}

#[inline]
pub fn get_layout_flex_directions<'a>(styled_dom: &StyledDom) -> NodeDataContainer<LayoutFlexDirection> {
    let cache = styled_dom.get_css_property_cache();
    let node_data_container = styled_dom.node_data.as_container();
    let styled_nodes = styled_dom.styled_nodes.as_container();
    assert!(node_data_container.internal.len() == styled_nodes.internal.len()); // elide bounds checking

    NodeDataContainer {
        internal: styled_nodes.internal.par_iter().enumerate().map(|(node_id, styled_node)| {
            cache.get_flex_direction(&node_data_container.internal[node_id], &NodeId::new(node_id), &styled_node.state).unwrap_or_default().get_property_or_default().unwrap_or_default()
        }).collect()
    }
}

#[inline]
pub fn get_layout_flex_grows<'a>(styled_dom: &StyledDom) -> NodeDataContainer<f32> {
    // Prevent flex-grow and flex-shrink to be less than 0
    let cache = styled_dom.get_css_property_cache();
    let node_data_container = styled_dom.node_data.as_container();
    let styled_nodes = styled_dom.styled_nodes.as_container();
    assert!(node_data_container.internal.len() == styled_nodes.internal.len()); // elide bounds checking

    NodeDataContainer {
        internal: styled_nodes.internal.par_iter().enumerate().map(|(node_id, styled_node)| {
            cache.get_flex_grow(&node_data_container.internal[node_id], &NodeId::new(node_id), &styled_node.state)
            .and_then(|g| g.get_property().copied())
            .and_then(|grow| Some(grow.inner.get().max(0.0)))
            .unwrap_or(DEFAULT_FLEX_GROW_FACTOR)
        }).collect()
    }
}

fn get_overflow(
    overflow_x: &Option<CssPropertyValue<LayoutOverflow>>,
    overflow_y: &Option<CssPropertyValue<LayoutOverflow>>,
    parent_rect: &LayoutRect,
    children_sum_rect: &Option<LayoutRect>
) -> OverflowInfo {

    use azul_core::ui_solver::DirectionalOverflowInfo;

    let overflow_x = overflow_x.and_then(|p| p.get_property_or_default()).unwrap_or_default();
    let overflow_y = overflow_y.and_then(|p| p.get_property_or_default()).unwrap_or_default();

    match children_sum_rect {
        Some(children_sum_rect) => {
            let overflow_x_amount = (parent_rect.size.width + parent_rect.origin.x) - (children_sum_rect.origin.x + children_sum_rect.size.width);
            let overflow_y_amount = (parent_rect.size.height + parent_rect.origin.y) - (children_sum_rect.origin.y + children_sum_rect.size.height);
            OverflowInfo {
                overflow_x: match overflow_x {
                    LayoutOverflow::Scroll => DirectionalOverflowInfo::Scroll { amount: Some(overflow_x_amount) },
                    LayoutOverflow::Auto => DirectionalOverflowInfo::Auto { amount: Some(overflow_x_amount) },
                    LayoutOverflow::Hidden => DirectionalOverflowInfo::Hidden { amount: Some(overflow_x_amount) },
                    LayoutOverflow::Visible => DirectionalOverflowInfo::Visible { amount: Some(overflow_x_amount) },
                },
                overflow_y: match overflow_y {
                    LayoutOverflow::Scroll => DirectionalOverflowInfo::Scroll { amount: Some(overflow_y_amount) },
                    LayoutOverflow::Auto => DirectionalOverflowInfo::Auto { amount: Some(overflow_y_amount) },
                    LayoutOverflow::Hidden => DirectionalOverflowInfo::Hidden { amount: Some(overflow_y_amount) },
                    LayoutOverflow::Visible => DirectionalOverflowInfo::Visible { amount: Some(overflow_y_amount) },
                }
            }
        },
        None => {
            OverflowInfo {
                overflow_x: match overflow_x {
                    LayoutOverflow::Scroll => DirectionalOverflowInfo::Scroll { amount: None },
                    LayoutOverflow::Auto => DirectionalOverflowInfo::Auto { amount: None },
                    LayoutOverflow::Hidden => DirectionalOverflowInfo::Hidden { amount: None },
                    LayoutOverflow::Visible => DirectionalOverflowInfo::Visible { amount: None },
                },
                overflow_y: match overflow_y {
                    LayoutOverflow::Scroll => DirectionalOverflowInfo::Scroll { amount: None },
                    LayoutOverflow::Auto => DirectionalOverflowInfo::Auto { amount: None },
                    LayoutOverflow::Hidden => DirectionalOverflowInfo::Hidden { amount: None },
                    LayoutOverflow::Visible => DirectionalOverflowInfo::Visible { amount: None },
                }
            }
        }
    }
}


fn precalculate_all_offsets(styled_dom: &StyledDom) -> NodeDataContainer<AllOffsets> {

    use rayon::prelude::*;

    let css_property_cache = styled_dom.get_css_property_cache();
    let node_data_container = styled_dom.node_data.as_container();
    let styled_nodes = styled_dom.styled_nodes.as_container();
    assert!(styled_nodes.internal.len() == node_data_container.internal.len()); // elide bounds check

    NodeDataContainer {
        internal: styled_nodes.internal
        .par_iter()
        .enumerate()
        .map(|(node_id_usize, styled_node)| {
            let node_id = NodeId::new(node_id_usize);
            let state = &styled_node.state;
            precalculate_offset(&node_data_container.internal[node_id_usize], &css_property_cache, &node_id, state)
        })
        .collect(),
    }
}

fn precalculate_offset(node_data: &NodeData, css_property_cache: &CssPropertyCache, node_id: &NodeId, state: &StyledNodeState) -> AllOffsets {
    AllOffsets {
        border_widths: LayoutBorderOffsets {
            left: css_property_cache.get_border_left_width(node_data, node_id, state),
            right: css_property_cache.get_border_right_width(node_data, node_id, state),
            top: css_property_cache.get_border_top_width(node_data, node_id, state),
            bottom: css_property_cache.get_border_bottom_width(node_data, node_id, state),
        },
        padding: LayoutPaddingOffsets {
            left: css_property_cache.get_padding_left(node_data, node_id, state),
            right: css_property_cache.get_padding_right(node_data, node_id, state),
            top: css_property_cache.get_padding_top(node_data, node_id, state),
            bottom: css_property_cache.get_padding_bottom(node_data, node_id, state),
        },
        margin: LayoutMarginOffsets {
            left: css_property_cache.get_margin_left(node_data, node_id, state),
            right: css_property_cache.get_margin_right(node_data, node_id, state),
            top: css_property_cache.get_margin_top(node_data, node_id, state),
            bottom: css_property_cache.get_margin_bottom(node_data, node_id, state),
        },
        position: LayoutAbsolutePositions {
            left: css_property_cache.get_left(node_data, node_id, state),
            right: css_property_cache.get_right(node_data, node_id, state),
            top: css_property_cache.get_top(node_data, node_id, state),
            bottom: css_property_cache.get_bottom(node_data, node_id, state),
        },
        overflow_x: css_property_cache.get_overflow_x(node_data, node_id, state),
        overflow_y: css_property_cache.get_overflow_y(node_data, node_id, state),
    }
}

struct AllOffsets {
    border_widths: LayoutBorderOffsets,
    padding: LayoutPaddingOffsets,
    margin: LayoutMarginOffsets,
    position: LayoutAbsolutePositions,
    overflow_x: Option<CssPropertyValue<LayoutOverflow>>,
    overflow_y: Option<CssPropertyValue<LayoutOverflow>>,
}

pub struct LayoutAbsolutePositions {
    left: Option<CssPropertyValue<LayoutLeft>>,
    right: Option<CssPropertyValue<LayoutRight>>,
    top: Option<CssPropertyValue<LayoutTop>>,
    bottom: Option<CssPropertyValue<LayoutBottom>>,
}

impl LayoutAbsolutePositions {
    pub fn resolve(&self, parent_scale_x: f32, parent_scale_y: f32) -> ResolvedOffsets {
        ResolvedOffsets {
            left: self.left.and_then(|p| Some(p.get_property()?.inner.to_pixels(parent_scale_x))).unwrap_or_default(),
            top: self.top.and_then(|p| Some(p.get_property()?.inner.to_pixels(parent_scale_y))).unwrap_or_default(),
            bottom: self.bottom.and_then(|p| Some(p.get_property()?.inner.to_pixels(parent_scale_y))).unwrap_or_default(),
            right: self.right.and_then(|p| Some(p.get_property()?.inner.to_pixels(parent_scale_x))).unwrap_or_default(),
        }
    }
}

pub struct LayoutBorderOffsets {
    left: Option<CssPropertyValue<LayoutBorderLeftWidth>>,
    right: Option<CssPropertyValue<LayoutBorderRightWidth>>,
    top: Option<CssPropertyValue<LayoutBorderTopWidth>>,
    bottom: Option<CssPropertyValue<LayoutBorderBottomWidth>>,
}

impl LayoutBorderOffsets {
    pub fn resolve(&self, parent_scale_x: f32, parent_scale_y: f32) -> ResolvedOffsets {
        ResolvedOffsets {
            left: self.left.and_then(|p| Some(p.get_property()?.inner.to_pixels(parent_scale_x))).unwrap_or_default(),
            top: self.top.and_then(|p| Some(p.get_property()?.inner.to_pixels(parent_scale_y))).unwrap_or_default(),
            bottom: self.bottom.and_then(|p| Some(p.get_property()?.inner.to_pixels(parent_scale_y))).unwrap_or_default(),
            right: self.right.and_then(|p| Some(p.get_property()?.inner.to_pixels(parent_scale_x))).unwrap_or_default(),
        }
    }
}

pub struct LayoutPaddingOffsets {
    left: Option<CssPropertyValue<LayoutPaddingLeft>>,
    right: Option<CssPropertyValue<LayoutPaddingRight>>,
    top: Option<CssPropertyValue<LayoutPaddingTop>>,
    bottom: Option<CssPropertyValue<LayoutPaddingBottom>>,
}

impl LayoutPaddingOffsets {
    pub fn resolve(&self, parent_scale_x: f32, parent_scale_y: f32) -> ResolvedOffsets {
        ResolvedOffsets {
            left: self.left.and_then(|p| Some(p.get_property()?.inner.to_pixels(parent_scale_x))).unwrap_or_default(),
            top: self.top.and_then(|p| Some(p.get_property()?.inner.to_pixels(parent_scale_y))).unwrap_or_default(),
            bottom: self.bottom.and_then(|p| Some(p.get_property()?.inner.to_pixels(parent_scale_y))).unwrap_or_default(),
            right: self.right.and_then(|p| Some(p.get_property()?.inner.to_pixels(parent_scale_x))).unwrap_or_default(),
        }
    }
}

struct LayoutMarginOffsets {
    left: Option<CssPropertyValue<LayoutMarginLeft>>,
    right: Option<CssPropertyValue<LayoutMarginRight>>,
    top: Option<CssPropertyValue<LayoutMarginTop>>,
    bottom: Option<CssPropertyValue<LayoutMarginBottom>>,
}

impl LayoutMarginOffsets {
    pub fn resolve(&self, parent_scale_x: f32, parent_scale_y: f32) -> ResolvedOffsets {
        ResolvedOffsets {
            left: self.left.and_then(|p| Some(p.get_property()?.inner.to_pixels(parent_scale_x))).unwrap_or_default(),
            top: self.top.and_then(|p| Some(p.get_property()?.inner.to_pixels(parent_scale_y))).unwrap_or_default(),
            bottom: self.bottom.and_then(|p| Some(p.get_property()?.inner.to_pixels(parent_scale_y))).unwrap_or_default(),
            right: self.right.and_then(|p| Some(p.get_property()?.inner.to_pixels(parent_scale_x))).unwrap_or_default(),
        }
    }
}

// Adds the image and font resources to the app_resources but does NOT add them to the RenderAPI
pub fn do_the_layout(
    styled_dom: StyledDom,
    app_resources: &mut AppResources,
    fc_cache: &FcFontCache,
    all_resource_updates: &mut Vec<ResourceUpdate>,
    id_namespace: IdNamespace,
    pipeline_id: PipelineId,
    callbacks: RenderCallbacks,
    full_window_state: &FullWindowState,
) -> Vec<LayoutResult> {

    let mut current_dom_id = 0;
    let mut doms = vec![
        (None, DomId { inner: current_dom_id }, styled_dom, LogicalRect::new(LogicalPosition::zero(), full_window_state.size.dimensions)),
    ];
    let mut resolved_doms = Vec::new();

    loop {

        let mut new_doms = Vec::new();

        for (parent_dom_id, dom_id, styled_dom, rect) in doms.drain(..) {

            use azul_core::app_resources::add_fonts_and_images;

            add_fonts_and_images(
                app_resources,
                fc_cache,
                id_namespace,
                all_resource_updates,
                &pipeline_id,
                &styled_dom,
                callbacks.load_font_fn,
                callbacks.load_image_fn,
                callbacks.parse_font_fn,
            );

            let mut layout_result = do_the_layout_internal(
                dom_id,
                parent_dom_id,
                styled_dom,
                app_resources,
                pipeline_id,
                rect,
            );


            let mut iframe_mapping = BTreeMap::new();

            for iframe_node_id in layout_result.styled_dom.scan_for_iframe_callbacks() {

                use azul_core::callbacks::{HidpiAdjustedBounds, IFrameCallbackInfo, IFrameCallbackReturn};

                // Generate a new DomID
                current_dom_id += 1;
                let iframe_dom_id = DomId { inner: current_dom_id };
                iframe_mapping.insert(iframe_node_id, iframe_dom_id);

                let bounds = &layout_result.rects.as_ref()[iframe_node_id];
                let bounds_size = LayoutSize::new(bounds.size.width.round() as isize, bounds.size.height.round() as isize);
                let hidpi_bounds = HidpiAdjustedBounds::from_bounds(bounds_size, full_window_state.size.hidpi_factor);

                // Invoke the IFrame callback
                let iframe_return: IFrameCallbackReturn = {

                    let iframe_callback_info = IFrameCallbackInfo::new(
                        &app_resources,
                        hidpi_bounds,

                        // TODO - see /examples/assets/images/scrollbounds.png for documentation!
                        /* scroll_size  */ bounds.size,
                        /* scroll_offset */ LogicalPosition::zero(),
                        /* virtual_scroll_size  */ bounds.size,
                        /* virtual_scroll_offset */ LogicalPosition::zero(),
                    );

                    let mut node_data_mut = layout_result.styled_dom.node_data.as_container_mut();
                    match &mut node_data_mut[iframe_node_id].get_iframe_node() {
                        Some(iframe_node) => {
                            (iframe_node.callback.cb)(&mut iframe_node.data, iframe_callback_info)
                        },
                        None => IFrameCallbackReturn::default(),
                    }
                };

                let mut iframe_dom = iframe_return.dom;

                // TODO: use other fields of iframe_return here!

                let hovered_nodes = full_window_state.hovered_nodes.get(&iframe_dom_id).map(|i| i.regular_hit_test_nodes.clone()).unwrap_or_default().keys().cloned().collect::<Vec<_>>();
                let active_nodes = if !full_window_state.mouse_state.mouse_down() { Vec::new() } else { hovered_nodes.clone() };

                let _ = iframe_dom.restyle_nodes_hover(hovered_nodes.as_slice(), true);
                let _ = iframe_dom.restyle_nodes_active(active_nodes.as_slice(), true);
                if let Some(focused_node) = full_window_state.focused_node {
                    if focused_node.dom == iframe_dom_id {
                        let _ = iframe_dom.restyle_nodes_focus(&[focused_node.node.into_crate_internal().unwrap()], true);
                    }
                }

                let bounds = LogicalRect::new(LogicalPosition::zero(), hidpi_bounds.get_logical_size()); // TODO: use the iframe static position here?
                // push the styled iframe dom into the next iframes and repeat (recurse)
                new_doms.push((Some(dom_id), iframe_dom_id, iframe_dom, bounds));
            }

            layout_result.iframe_mapping = iframe_mapping;
            resolved_doms.push(layout_result);
        }

        if new_doms.is_empty() {
            break;
        } else {
            doms = new_doms;
        }
    }

    resolved_doms
}

/// At this point in time, all font keys, image keys, etc. have to be already
/// been submitted to the RenderApi and the AppResources!
pub fn do_the_layout_internal(
    dom_id: DomId,
    parent_dom_id: Option<DomId>,
    styled_dom: StyledDom,
    app_resources: &mut AppResources,
    pipeline_id: PipelineId,
    bounds: LogicalRect
) -> LayoutResult {

    let rect_size = bounds.size;
    let rect_offset = bounds.origin;

    // TODO: Filter all inline text blocks: inline blocks + their padding + margin
    // The NodeId has to be the **next** NodeId (the next sibling after the inline element)
    // let mut inline_text_blocks = BTreeMap::<NodeId, InlineText>::new();

    let all_parents_btreeset = styled_dom.non_leaf_nodes.iter().filter_map(|p| Some(p.node_id.into_crate_internal()?)).collect::<BTreeSet<_>>();
    let all_nodes_btreeset = (0..styled_dom.node_data.as_container().len()).map(|n| NodeId::new(n)).collect::<BTreeSet<_>>();

    let layout_position_info = get_layout_positions(&styled_dom);
    let layout_flex_grow_info = get_layout_flex_grows(&styled_dom);
    let layout_directions_info = get_layout_flex_directions(&styled_dom);
    let layout_justify_contents = get_layout_justify_contents(&styled_dom);
    let layout_offsets = precalculate_all_offsets(&styled_dom);
    let layout_width_heights = precalculate_wh_config(&styled_dom);

    let content_widths_pre = NodeDataContainer { internal: vec![None; styled_dom.node_data.len()] };

    let mut width_calculated_arena = width_calculated_rect_arena_from_rect_layout_arena(
        &layout_width_heights.as_ref(),
        &layout_offsets.as_ref(),
        &content_widths_pre.as_ref(),
        &styled_dom.node_hierarchy.as_container(),
        &styled_dom.non_leaf_nodes.as_ref(),
        rect_size.width,
    );

    solve_flex_layout_width(
        &mut width_calculated_arena.as_ref_mut(),
        &layout_flex_grow_info.as_ref(),
        &layout_position_info.as_ref(),
        &layout_directions_info.as_ref(),
        &styled_dom.node_hierarchy.as_container(),
        styled_dom.non_leaf_nodes.as_ref(),
        rect_size.width,
        &all_parents_btreeset,
    );

    // Break all strings into words and / or resolve the TextIds
    let word_cache = {
        #[cfg(feature = "text_layout")] {
            create_word_cache(&styled_dom.node_data.as_container())
        }
        #[cfg(not(feature = "text_layout"))] {
            BTreeMap::new()
        }
    };
    // Scale the words to the correct size - TODO: Cache this in the app_resources!
    let shaped_words = {
        #[cfg(feature = "text_layout")] {
            create_shaped_words(&pipeline_id, app_resources, &word_cache, &styled_dom)
        }
        #[cfg(not(feature = "text_layout"))] {
            BTreeMap::new()
        }
    };

    // Layout all words as if there was no max-width constraint (to get the texts "content width").
    let mut word_positions_no_max_width = BTreeMap::new();
    #[cfg(feature = "text_layout")]
    create_word_positions(
        &mut word_positions_no_max_width,
        &all_nodes_btreeset,
        &pipeline_id,
        app_resources,
        &word_cache,
        &shaped_words,
        &styled_dom,
        &width_calculated_arena.as_ref(),
    );

    // Determine the preferred **content** width
    // let content_widths = node_data.transform(|node, node_id| {
    //     get_content_width(pipeline_id, &node_id, &node.get_node_type(), app_resources, &word_positions_no_max_width)
    // });

    // // Layout the words again, this time with the proper width constraints!
    // let proper_max_widths = solved_widths.solved_widths.linear_iter().map(|node_id| {
    //     (node_id, TextSizePx(solved_widths.solved_widths[node_id].total()))
    // }).collect();

    // let word_positions_with_max_width = create_word_positions(&word_cache, &shaped_words, display_rects, &proper_max_widths, &inline_text_blocks);

    // Get the content height of the content
    // let content_heights = node_data.transform(|node, node_id| {
    //     let div_width = solved_widths.solved_widths[node_id].total();
    //     get_content_height(pipeline_id, &node_id, &node.get_node_type(), app_resources, &word_positions_no_max_width, div_width)
    // });

    let content_heights_pre = NodeDataContainer { internal: vec![None; styled_dom.node_data.len()] };

    // TODO: The content height is not the final height!
    let mut height_calculated_arena = height_calculated_rect_arena_from_rect_layout_arena(
        &layout_width_heights.as_ref(),
        &layout_offsets.as_ref(),
        &content_heights_pre.as_ref(),
        &styled_dom.node_hierarchy.as_container(),
        &styled_dom.non_leaf_nodes.as_ref(),
        rect_size.height,
    );

    solve_flex_layout_height(
        &mut height_calculated_arena.as_ref_mut(),
        &layout_flex_grow_info.as_ref(),
        &layout_position_info.as_ref(),
        &layout_directions_info.as_ref(),
        &styled_dom.node_hierarchy.as_container(),
        styled_dom.non_leaf_nodes.as_ref(),
        rect_size.height,
        &all_parents_btreeset,
    );

    let mut x_positions = NodeDataContainer::new(vec![HorizontalSolvedPosition(0.0); styled_dom.node_data.as_ref().len()]);
    get_x_positions(
        &mut x_positions.as_ref_mut(),
        &width_calculated_arena.as_ref(),
        &styled_dom.node_hierarchy.as_container(),
        &layout_position_info.as_ref(),
        &layout_directions_info.as_ref(),
        &layout_justify_contents.as_ref(),
        &styled_dom.non_leaf_nodes.as_ref(),
        rect_offset.clone(),
        &all_parents_btreeset,
    );

    let mut y_positions = NodeDataContainer::new(vec![VerticalSolvedPosition(0.0); styled_dom.node_data.as_ref().len()]);
    get_y_positions(
        &mut y_positions.as_ref_mut(),
        &height_calculated_arena.as_ref(),
        &styled_dom.node_hierarchy.as_container(),
        &layout_position_info.as_ref(),
        &layout_directions_info.as_ref(),
        &layout_justify_contents.as_ref(),
        &styled_dom.non_leaf_nodes.as_ref(),
        rect_offset,
        &all_parents_btreeset,
    );

    let mut positioned_rects = NodeDataContainer { internal: vec![PositionedRectangle::default(); styled_dom.node_data.len()].into() };
    let nodes_that_updated_positions = all_nodes_btreeset.clone();
    let nodes_that_need_to_redraw_text = all_nodes_btreeset.clone();

    {
        let layout_offsets_ref = layout_offsets.as_ref();
        position_nodes(
            &mut positioned_rects.as_ref_mut(),
            &styled_dom,
            AllOffsetsProvider::All(&layout_offsets_ref),
            &width_calculated_arena.as_ref(),
            &height_calculated_arena.as_ref(),
            &x_positions.as_ref(),
            &y_positions.as_ref(),
            &nodes_that_updated_positions,
            &nodes_that_need_to_redraw_text,
            &layout_position_info.as_ref(),
            &word_cache,
            &shaped_words,
            &word_positions_no_max_width,
            pipeline_id
        );
    }

    let mut overflowing_rects = ScrolledNodes::default();
    get_nodes_that_need_scroll_clip(
        &mut overflowing_rects,
        &styled_dom.styled_nodes.as_container(),
        &styled_dom.node_data.as_container(),
        &positioned_rects.as_ref(),
        styled_dom.non_leaf_nodes.as_ref(),
        pipeline_id,
    );

    LayoutResult {
        dom_id,
        parent_dom_id,
        styled_dom,
        root_size: LayoutSize::new(rect_size.width.round() as isize, rect_size.height.round() as isize),
        root_position: LayoutPoint::new(rect_offset.x.round() as isize, rect_offset.y.round() as isize),
        preferred_widths: content_widths_pre,
        preferred_heights: content_heights_pre,
        width_calculated_rects: width_calculated_arena,
        height_calculated_rects: height_calculated_arena,
        solved_pos_x: x_positions,
        solved_pos_y: y_positions,
        layout_flex_grows: layout_flex_grow_info,
        layout_positions: layout_position_info,
        layout_flex_directions: layout_directions_info,
        layout_justify_contents: layout_justify_contents,
        rects: positioned_rects,
        words_cache: word_cache,
        shaped_words_cache: shaped_words,
        positioned_words_cache: word_positions_no_max_width,
        scrollable_nodes: overflowing_rects,
        iframe_mapping: BTreeMap::new(),
    }
}

/// Note: because this function is called both on layout() and relayout(),
/// the offsets are calculated during the layout() run. However,
/// we don't want to store all offsets because that would waste memory
///
/// So you can EITHER specify all offsets (useful during layout()) or specify
/// only the offsets of nodes that need to be recalculated (useful during relayout())
///
/// If an offset isn't found (usually shouldn't happen), the final positioned
/// rectangle is not positioned.
enum AllOffsetsProvider<'a> {
    All(&'a NodeDataContainerRef<'a, AllOffsets>),
    OnlyRecalculatedNodes(&'a BTreeMap<NodeId, AllOffsets>),
}

impl<'a> AllOffsetsProvider<'a> {
    fn get_offsets_for_node(&self, node_id: &NodeId) -> Option<&AllOffsets> {
        match self {
            AllOffsetsProvider::All(a) => Some(&a[*node_id]),
            AllOffsetsProvider::OnlyRecalculatedNodes(b) => b.get(node_id)
        }
    }
}

fn position_nodes<'a>(
    positioned_rects: &mut NodeDataContainerRefMut<'a, PositionedRectangle>,
    styled_dom: &StyledDom,
    offsets: AllOffsetsProvider<'a>,
    solved_widths: &NodeDataContainerRef<'a, WidthCalculatedRect>,
    solved_heights: &NodeDataContainerRef<'a, HeightCalculatedRect>,
    x_positions: &NodeDataContainerRef<'a, HorizontalSolvedPosition>,
    y_positions: &NodeDataContainerRef<'a, VerticalSolvedPosition>,
    nodes_that_updated_positions: &BTreeSet<NodeId>,
    nodes_that_need_to_redraw_text: &BTreeSet<NodeId>,
    position_info: &NodeDataContainerRef<'a, LayoutPosition>,
    word_cache: &BTreeMap<NodeId, Words>,
    shaped_words: &BTreeMap<NodeId, ShapedWords>,
    word_positions: &BTreeMap<NodeId, (WordPositions, FontInstanceKey)>,
    pipeline_id: PipelineId,
) {

    use azul_core::ui_solver::PositionInfo;

    let mut positioned_node_stack = vec![NodeId::new(0)];
    let css_property_cache = styled_dom.get_css_property_cache();
    let styled_nodes = styled_dom.styled_nodes.as_container();

    // create the final positioned rectangles
    for ParentWithNodeDepth { depth: _, node_id } in styled_dom.non_leaf_nodes.as_ref().iter() {

        let parent_node_id = match node_id.into_crate_internal() { Some(s) => s, None => continue, };
        if !nodes_that_updated_positions.contains(&parent_node_id) { continue; };
        let parent_node_data = &styled_dom.node_data.as_container()[parent_node_id];

        let parent_position = position_info[parent_node_id];
        let width = solved_widths[parent_node_id];
        let height = solved_heights[parent_node_id];
        let x_pos = x_positions[parent_node_id].0;
        let y_pos = y_positions[parent_node_id].0;
        let parent_styled_node_state = &styled_nodes[parent_node_id].state;

        let parent_parent_node_id = styled_dom.node_hierarchy.as_container()[parent_node_id].parent_id().unwrap_or(NodeId::new(0));
        let parent_x_pos = x_positions[parent_parent_node_id].0;
        let parent_y_pos = y_positions[parent_parent_node_id].0;
        let parent_parent_width = solved_widths[parent_parent_node_id];
        let parent_parent_height = solved_heights[parent_parent_node_id];

        let last_positioned_item_node_id = positioned_node_stack.last().map(|l| *l).unwrap_or(NodeId::new(0));
        let last_positioned_item_x_pos = x_positions[last_positioned_item_node_id].0;
        let last_positioned_item_y_pos = y_positions[last_positioned_item_node_id].0;
        let parent_position_info = match parent_position {
            LayoutPosition::Static => PositionInfo::Static {
                // calculate relative to parent
                x_offset: x_pos - parent_x_pos,
                y_offset: y_pos - parent_y_pos,
                static_x_offset: x_pos,
                static_y_offset: y_pos,
            },
            LayoutPosition::Relative => PositionInfo::Relative {
                // calculate relative to parent
                x_offset: x_pos - parent_x_pos,
                y_offset: y_pos - parent_y_pos,
                static_x_offset: x_pos,
                static_y_offset: y_pos,
            },
            LayoutPosition::Absolute => PositionInfo::Absolute {
                // calculate relative to last positioned item
                x_offset: x_pos - last_positioned_item_x_pos,
                y_offset: y_pos - last_positioned_item_y_pos,
                static_x_offset: x_pos,
                static_y_offset: y_pos,
            },
            LayoutPosition::Fixed => PositionInfo::Fixed {
                // relative to screen, already done
                x_offset: x_pos,
                y_offset: y_pos,
                static_x_offset: x_pos,
                static_y_offset: y_pos,
            },
        };
        let parent_size = LogicalSize::new(width.total(), height.total());

        let parent_offsets = match offsets.get_offsets_for_node(&parent_node_id) {
            Some(s) => s,
            None => continue,
        };

        let parent_padding = parent_offsets.padding.resolve(parent_parent_width.total(), parent_parent_height.total());
        let parent_margin = parent_offsets.margin.resolve(parent_parent_width.total(), parent_parent_height.total());
        let parent_border_widths = parent_offsets.border_widths.resolve(parent_parent_width.total(), parent_parent_height.total());
        let parent_parent_size = LogicalSize::new(parent_parent_width.total(), parent_parent_height.total());

        let parent_sum_rect = LayoutRect::new(
            LayoutPoint::new(x_pos.round() as isize, y_pos.round() as isize),
            LayoutSize::new(width.total().round() as isize, height.total().round() as isize),
        );
        let mut children_sum_rects = Vec::new();

        // push positioned item and layout children
        if parent_position != LayoutPosition::Static {
            positioned_node_stack.push(parent_node_id);
        }

        for child_node_id in parent_node_id.az_children(&styled_dom.node_hierarchy.as_container()) {

            // copy the width and height from the parent node
            let parent_width = width;
            let parent_height = height;
            let parent_x_pos = x_pos;
            let parent_y_pos = y_pos;

            let width = solved_widths[child_node_id];
            let height = solved_heights[child_node_id];
            let x_pos = x_positions[child_node_id].0;
            let y_pos = y_positions[child_node_id].0;
            let child_position = position_info[child_node_id];
            let child_styled_node_state = &styled_nodes[child_node_id].state;
            let child_node_data = &styled_dom.node_data.as_container()[child_node_id];

            let child_position = match child_position {
                LayoutPosition::Static => PositionInfo::Static {
                    // calculate relative to parent
                    x_offset: x_pos - parent_x_pos,
                    y_offset: y_pos - parent_y_pos,
                    static_x_offset: x_pos,
                    static_y_offset: y_pos,
                },
                LayoutPosition::Relative => PositionInfo::Relative {
                    // calculate relative to parent
                    x_offset: x_pos - parent_x_pos,
                    y_offset: y_pos - parent_y_pos,
                    static_x_offset: x_pos,
                    static_y_offset: y_pos,
                },
                LayoutPosition::Absolute => PositionInfo::Absolute {
                    // calculate relative to last positioned item
                    x_offset: x_pos - last_positioned_item_x_pos,
                    y_offset: y_pos - last_positioned_item_y_pos,
                    static_x_offset: x_pos,
                    static_y_offset: y_pos,
                },
                LayoutPosition::Fixed => PositionInfo::Fixed {
                    // relative to screen, already done
                    x_offset: x_pos,
                    y_offset: y_pos,
                    static_x_offset: x_pos,
                    static_y_offset: y_pos,
                },
            };

            let parent_size = LogicalSize::new(parent_width.total(), parent_height.total());
            let child_size = LayoutSize::new(width.total().round() as isize, height.total().round() as isize);
            let child_rect = LayoutRect::new(LayoutPoint::new(x_pos.round() as isize, y_pos.round() as isize), child_size);

            children_sum_rects.push(child_rect);

            let child_offsets = match offsets.get_offsets_for_node(&child_node_id) {
                Some(s) => s,
                None => continue,
            };

            let child_padding = child_offsets.padding.resolve(parent_width.total(), parent_height.total());
            let child_margin = child_offsets.margin.resolve(parent_width.total(), parent_height.total());
            let child_border_widths = child_offsets.border_widths.resolve(parent_width.total(), parent_height.total());

            // set text, if any
            let child_text = if let (
                Some(words),
                Some(shaped_words),
                Some((word_positions, _))
            ) = (
                word_cache.get(&child_node_id),
                shaped_words.get(&child_node_id),
                word_positions.get(&child_node_id)
            ) {
                if nodes_that_need_to_redraw_text.contains(&child_node_id) {
                    #[cfg(feature = "text_layout")] {
                        use azul_text_layout::InlineText;
                        let mut inline_text_layout = InlineText { words, shaped_words }.get_text_layout(pipeline_id, child_node_id, &word_positions.text_layout_options);
                        let (horz_alignment, vert_alignment) = determine_text_alignment(
                            css_property_cache.get_align_items(child_node_data, &child_node_id, child_styled_node_state),
                            css_property_cache.get_justify_content(child_node_data, &child_node_id, child_styled_node_state),
                            css_property_cache.get_text_align(child_node_data, &child_node_id, child_styled_node_state),
                        );
                        inline_text_layout.align_children_horizontal(horz_alignment);
                        inline_text_layout.align_children_vertical_in_parent_bounds(&parent_size, vert_alignment);
                        Some((word_positions.text_layout_options.clone(), inline_text_layout))
                    }
                    #[cfg(not(feature = "text_layout"))] {
                        None
                    }
                } else {
                    positioned_rects[child_node_id].resolved_text_layout_options.clone()
                }
            } else {
                None
            };

            let child_overflow = get_overflow(&child_offsets.overflow_x, &child_offsets.overflow_y, &child_rect, &None);

            positioned_rects[child_node_id] = PositionedRectangle {
                size: LogicalSize::new(width.total(), height.total()),
                position: child_position,
                padding: child_padding,
                margin: child_margin,
                border_widths: child_border_widths,
                resolved_text_layout_options: child_text,
                overflow: child_overflow,
            };
        }

        let children_sum_rect = LayoutRect::union(children_sum_rects.into_iter());
        let parent_overflow = get_overflow(&parent_offsets.overflow_x, &parent_offsets.overflow_y, &parent_sum_rect, &children_sum_rect);

        // NOTE: Intentionally do not set text_layout_options,
        // otherwise this would overwrite the existing text layout options
        // Label / Text nodes are ALWAYS children of some parent node,
        // they can not be root nodes. Therefore the children_iter() will take
        // care of layouting the text
        let parent_rect = &mut positioned_rects[parent_node_id];
        parent_rect.size = parent_size;
        parent_rect.position = parent_position_info;
        parent_rect.padding = parent_padding;
        parent_rect.margin = parent_margin;
        parent_rect.border_widths = parent_border_widths;
        parent_rect.overflow = parent_overflow;

        if parent_position != LayoutPosition::Static {
            positioned_node_stack.pop();
        }
    }
}

#[cfg(feature = "text_layout")]
fn create_word_cache<'a>(
    node_data: &NodeDataContainerRef<'a, NodeData>,
) -> BTreeMap<NodeId, Words>
{
    use azul_text_layout::text_layout::split_text_into_words;

    let word_map = node_data.internal.par_iter().enumerate().map(|(node_id, node)| {
        let node_id = NodeId::new(node_id);
        match node.get_node_type() {
            NodeType::Label(string) => Some((node_id, split_text_into_words(string.as_str()))),
            _ => None,
        }
    }).collect::<Vec<_>>();

    word_map.into_iter().filter_map(|a| a).collect()
}

#[cfg(feature = "text_layout")]
pub fn create_shaped_words<'a>(
    pipeline_id: &PipelineId,
    app_resources: &mut AppResources,
    words: &BTreeMap<NodeId, Words>,
    styled_dom: &'a StyledDom,
) -> BTreeMap<NodeId, ShapedWords> {

    use azul_core::app_resources::ImmediateFontId;
    use azul_text_layout::text_layout::shape_words;

    let css_property_cache = styled_dom.get_css_property_cache();
    let styled_nodes = styled_dom.styled_nodes.as_container();
    let node_data = styled_dom.node_data.as_container();

    words
    .iter()
    .filter_map(|(node_id, words)| {

        let styled_node_state = &styled_nodes[*node_id].state;
        let node_data = &node_data[*node_id];
        let css_font_ids = css_property_cache.get_font_id_or_default(node_data, node_id, styled_node_state);
        let font_id = match app_resources.get_css_font_id(&css_font_ids) {
            Some(s) => ImmediateFontId::Resolved(*s),
            None => ImmediateFontId::Unresolved(css_font_ids),
        };

        let loaded_font = app_resources.get_loaded_font_mut(pipeline_id, &font_id)?;

        // downcast the loaded_font.font from Box<dyn Any> to Box<azul_text_layout::text_shaping::ParsedFont>
        let parsed_font_downcasted = loaded_font.font.downcast_mut::<azul_text_layout::text_shaping::ParsedFont>()?;

        let shaped_words = shape_words(words, parsed_font_downcasted);

        Some((*node_id, shaped_words))
    }).collect()
}

#[cfg(feature = "text_layout")]
fn create_word_positions<'a>(
    word_positions: &mut BTreeMap<NodeId, (WordPositions, FontInstanceKey)>,
    word_positions_to_generate: &BTreeSet<NodeId>,
    pipeline_id: &PipelineId,
    app_resources: &AppResources,
    words: &BTreeMap<NodeId, Words>,
    shaped_words: &BTreeMap<NodeId, ShapedWords>,
    styled_dom: &'a StyledDom,
    solved_widths: &'a NodeDataContainerRef<'a, WidthCalculatedRect>,
) {

    use azul_text_layout::text_layout::position_words;
    use azul_core::ui_solver::{ResolvedTextLayoutOptions, DEFAULT_LETTER_SPACING, DEFAULT_WORD_SPACING};
    use azul_core::app_resources::{ImmediateFontId, font_size_to_au};

    let css_property_cache = styled_dom.get_css_property_cache();
    let node_data_container = styled_dom.node_data.as_container();

    let collected =
    words
    .par_iter() // TODO: par_iter()!
    .filter_map(|(node_id, words)| {

        if !word_positions_to_generate.contains(node_id) { return None; }
        let node_data = &node_data_container[*node_id];

        let styled_node_state = styled_dom.get_styled_node_state(node_id);
        let font_size = css_property_cache.get_font_size_or_default(node_data, node_id, &styled_node_state);
        let font_size_au = font_size_to_au(font_size);
        let font_size_px = font_size.inner.to_pixels(DEFAULT_FONT_SIZE_PX as f32);
        let css_font_ids = css_property_cache.get_font_id_or_default(node_data, node_id, &styled_node_state);

        let font_id = match app_resources.get_css_font_id(&css_font_ids) {
            Some(s) => ImmediateFontId::Resolved(*s),
            None => ImmediateFontId::Unresolved(css_font_ids),
        };
        let loaded_font = app_resources.get_loaded_font(pipeline_id, &font_id)?;
        let font_instance_key = loaded_font.font_instances.get(&font_size_au)?;

        let shaped_words = shaped_words.get(&node_id)?;

        let text_can_overflow =  css_property_cache.get_overflow_x(node_data, node_id, &styled_node_state).unwrap_or_default().get_property_or_default().unwrap_or_default() != LayoutOverflow::Auto;
        let letter_spacing = css_property_cache.get_letter_spacing(node_data, node_id, &styled_node_state).and_then(|ls| Some(ls.get_property()?.inner.to_pixels(DEFAULT_LETTER_SPACING)));
        let word_spacing = css_property_cache.get_word_spacing(node_data, node_id, &styled_node_state).and_then(|ws| Some(ws.get_property()?.inner.to_pixels(DEFAULT_WORD_SPACING)));
        let line_height = css_property_cache.get_line_height(node_data, node_id, &styled_node_state).and_then(|lh| Some(lh.get_property()?.inner.get()));
        let tab_width = css_property_cache.get_tab_width(node_data, node_id, &styled_node_state).and_then(|tw| Some(tw.get_property()?.inner.get()));

        let text_layout_options = ResolvedTextLayoutOptions {
            max_horizontal_width: if text_can_overflow {
                Some(solved_widths[*node_id].total() as f32)
            } else {
                None
            }.into(),
            leading: None.into(), // TODO
            holes: Vec::new().into(), // TODO
            font_size_px,
            word_spacing: word_spacing.into(),
            letter_spacing: letter_spacing.into(),
            line_height: line_height.into(),
            tab_width: tab_width.into(),
        };

        let w = position_words(words, shaped_words, &text_layout_options);

        Some((*node_id, (w, *font_instance_key)))
    }).collect::<Vec<_>>();

    collected.into_iter().for_each(|(node_id, word_position)| {
        word_positions.insert(node_id, word_position);
    });
}

/// For a given rectangle, determines what text alignment should be used
fn determine_text_alignment(
    align_items: Option<CssPropertyValue<LayoutAlignItems>>,
    justify_content: Option<CssPropertyValue<LayoutJustifyContent>>,
    text_align: Option<CssPropertyValue<StyleTextAlignmentHorz>>,
)
    -> (StyleTextAlignmentHorz, StyleTextAlignmentVert)
{
    let mut horz_alignment = StyleTextAlignmentHorz::default();
    let mut vert_alignment = StyleTextAlignmentVert::default();

    if let Some(align_items) = align_items.as_ref() {
        // Vertical text alignment
        match align_items.get_property_or_default().unwrap_or_default() {
            LayoutAlignItems::FlexStart => vert_alignment = StyleTextAlignmentVert::Top,
            LayoutAlignItems::FlexEnd => vert_alignment = StyleTextAlignmentVert::Bottom,
            // technically stretch = blocktext, but we don't have that yet
            _ => vert_alignment = StyleTextAlignmentVert::Center,
        }
    }

    if let Some(justify_content) = justify_content.as_ref() {
        // Horizontal text alignment
        match justify_content.get_property_or_default().unwrap_or_default() {
            LayoutJustifyContent::Start => horz_alignment = StyleTextAlignmentHorz::Left,
            LayoutJustifyContent::End => horz_alignment = StyleTextAlignmentHorz::Right,
            _ => horz_alignment = StyleTextAlignmentHorz::Center,
        }
    }

    if let Some(text_align) = text_align.as_ref().and_then(|ta| ta.get_property().copied()) {
        // Horizontal text alignment with higher priority
        horz_alignment = text_align;
    }

    (horz_alignment, vert_alignment)
}


/// Returns all node IDs where the children overflow the parent, together with the
/// `(parent_rect, child_rect)` - the child rect is the sum of the children.
///
/// TODO: The performance of this function can be theoretically improved:
///
/// - Unioning the rectangles is heavier than just looping through the children and
/// summing up their width / height / padding + margin.
/// - Scroll nodes only need to be inserted if the parent doesn't have `overflow: hidden`
/// activated
/// - Overflow for X and Y needs to be tracked seperately (for overflow-x / overflow-y separation),
/// so there we'd need to track in which direction the inner_rect is overflowing.
fn get_nodes_that_need_scroll_clip(
    scrolled_nodes: &mut ScrolledNodes,
    display_list_rects: &NodeDataContainerRef<StyledNode>,
    dom_rects: &NodeDataContainerRef<NodeData>,
    layouted_rects: &NodeDataContainerRef<PositionedRectangle>,
    parents: &[ParentWithNodeDepth],
    pipeline_id: PipelineId,
) {

    use azul_core::ui_solver::{DirectionalOverflowInfo, OverflowingScrollNode, ExternalScrollId};
    use azul_core::dom::ScrollTagId;

    let mut nodes = BTreeMap::new();
    let mut tags_to_node_ids = BTreeMap::new();

    for ParentWithNodeDepth { depth: _, node_id } in parents.iter() {

        let parent_id = match node_id.into_crate_internal() { Some(s) => s, None => continue, };
        let parent_rect = &layouted_rects[parent_id];
        let overflow_x = &parent_rect.overflow.overflow_x;
        let overflow_y = &parent_rect.overflow.overflow_y;

        let scroll_rect = if overflow_x.is_none() || overflow_x.is_negative() {
            // no overflow in horizontal direction
            if overflow_y.is_none() || overflow_y.is_negative() {
                // no overflow in both directions
                None
            } else {
                // overflow in y, but not in x direction
                match overflow_y {
                    DirectionalOverflowInfo::Scroll { amount: Some(s) } | DirectionalOverflowInfo::Auto { amount: Some(s) } => {
                        Some(LayoutRect::new(LayoutPoint::zero(), LayoutSize::new(parent_rect.size.width.round() as isize, parent_rect.size.height.round() as isize + *s)))
                    },
                    _ => None
                }
            }
        } else {
            if overflow_y.is_none() || overflow_y.is_negative() {
                // overflow in x, but not in y direction
                match overflow_x {
                    DirectionalOverflowInfo::Scroll { amount: Some(s) } | DirectionalOverflowInfo::Auto { amount: Some(s) } => {
                        Some(LayoutRect::new(LayoutPoint::zero(), LayoutSize::new(parent_rect.size.width.round() as isize + *s, parent_rect.size.height.round() as isize)))
                    },
                    _ => None
                }
            } else {
                // overflow in x and y direction
                match overflow_x {
                    DirectionalOverflowInfo::Scroll { amount: Some(q) } | DirectionalOverflowInfo::Auto { amount: Some(q) } => {
                        match overflow_y {
                            DirectionalOverflowInfo::Scroll { amount: Some(s) } | DirectionalOverflowInfo::Auto { amount: Some(s) } => {
                                Some(LayoutRect::new(LayoutPoint::zero(), LayoutSize::new(parent_rect.size.width.round() as isize + *q, parent_rect.size.height.round() as isize + *s)))
                            },
                            _ => None
                        }
                    },
                    _ => None
                }
            }
        };

        let scroll_rect = match scroll_rect {
            Some(s) => s,
            None => continue,
        };

        let parent_dom_hash = dom_rects[parent_id].calculate_node_data_hash();

        // Create an external scroll id. This id is required to preserve its
        // scroll state accross multiple frames.
        let parent_external_scroll_id  = ExternalScrollId(parent_dom_hash.0, pipeline_id);

        // Create a unique scroll tag for hit-testing
        let scroll_tag_id = match display_list_rects.get(parent_id).and_then(|node| node.tag_id.into_option().map(|t| t.into_crate_internal())) {
            Some(existing_tag) => ScrollTagId(existing_tag),
            None => {
                if let Some(existing_overflowing_scroll_node) = scrolled_nodes.overflowing_nodes.get(node_id) {
                    existing_overflowing_scroll_node.scroll_tag_id.clone()
                } else {
                    ScrollTagId::new()
                }
            },
        };

        tags_to_node_ids.insert(scroll_tag_id, *node_id);
        nodes.insert(*node_id, OverflowingScrollNode {
            child_rect: scroll_rect,
            parent_external_scroll_id,
            parent_dom_hash,
            scroll_tag_id,
        });
    }

    *scrolled_nodes = ScrolledNodes { overflowing_nodes: nodes, tags_to_node_ids };
}

/// Relayout function, takes an existing LayoutResult and adjusts it
/// so that only the nodes that need relayout are touched.
/// See `CallbacksToCall`
///
/// Returns a vec of node IDs that whose layout was changed
pub fn do_the_relayout(
    root_bounds: LayoutRect,
    layout_result: &mut LayoutResult,
    app_resources: &mut AppResources,
    pipeline_id: PipelineId,
    nodes_to_relayout: &BTreeMap<NodeId, Vec<ChangedCssProperty>>
) -> Vec<NodeId> {

    // shortcut: in most cases, the root size hasn't changed and there
    // are no nodes to relayout

    let root_size = root_bounds.size;
    let root_size_changed = root_bounds != layout_result.get_bounds();

    if !root_size_changed && nodes_to_relayout.is_empty() {
        return Vec::new();
    }

    // merge the nodes to relayout by type so that we don't relayout twice
    let nodes_to_relayout = nodes_to_relayout
    .iter()
    .filter_map(|(node_id, changed_properties)| {
        let mut properties = BTreeMap::new();

        for prop in changed_properties.iter() {
            let prop_type = prop.previous_prop.get_type();
            if prop_type.can_trigger_relayout() {
                properties.insert(prop_type, prop.clone());
            }
        }

        if properties.is_empty() {
            None
        } else {
            Some((*node_id, properties))
        }
    }).collect::<BTreeMap<NodeId, BTreeMap<CssPropertyType, ChangedCssProperty>>>();

    if !root_size_changed && nodes_to_relayout.is_empty() {
        return Vec::new();
    }

    // ---- step 1: recalc size

    // TODO: for now, the preferred_widths and preferred_widths is always None,
    // so the content width + height isn't taken into account. If that changes,
    // the new content size has to be calculated first!

    // TODO: changes to display, float and box-sizing property are ignored
    // TODO: changes to top, bottom, right, left property are ignored for now
    // TODO: changes to position: property are updated, but ignored for now

    // recalc(&mut layout_result.preferred_widths);

    // update the precalculated properties (position, flex-grow, flex-direction, justify-content)
    nodes_to_relayout.keys().for_each(|node_id| {
        let styled_node_state = &layout_result.styled_dom.styled_nodes.as_container()[*node_id].state;
        let css_property_cache = layout_result.styled_dom.get_css_property_cache();
        let node_data = &layout_result.styled_dom.node_data.as_container()[*node_id];

        let new_layout_position = css_property_cache.get_position(node_data, node_id, styled_node_state).and_then(|p| p.get_property().copied()).unwrap_or_default();
        layout_result.layout_positions.as_ref_mut()[*node_id] = new_layout_position;

        let new_flex_grow = css_property_cache.get_flex_grow(node_data, node_id, styled_node_state)
            .and_then(|g| g.get_property().copied())
            .and_then(|grow| Some(grow.inner.get().max(0.0)))
            .unwrap_or(DEFAULT_FLEX_GROW_FACTOR);
        layout_result.layout_flex_grows.as_ref_mut()[*node_id] = new_flex_grow;

        let new_flex_direction = css_property_cache.get_flex_direction(node_data, node_id, styled_node_state).and_then(|p| p.get_property().copied()).unwrap_or_default();
        layout_result.layout_flex_directions.as_ref_mut()[*node_id] = new_flex_direction;

        let new_justify_content = css_property_cache.get_justify_content(node_data, node_id, styled_node_state).and_then(|p| p.get_property().copied()).unwrap_or_default();
        layout_result.layout_justify_contents.as_ref_mut()[*node_id] = new_justify_content;
    });

    let mut parents_that_need_to_recalc_width_of_children = BTreeSet::new();
    let mut parents_that_need_to_recalc_height_of_children = BTreeSet::new();
    let mut nodes_that_need_to_bubble_width = BTreeMap::new();
    let mut nodes_that_need_to_bubble_height = BTreeMap::new();
    let mut parents_that_need_to_reposition_children_x = BTreeSet::new();
    let mut parents_that_need_to_reposition_children_y = BTreeSet::new();

    if root_size.width != layout_result.root_size.width {
        let root_id = layout_result.styled_dom.root.into_crate_internal().unwrap();
        parents_that_need_to_recalc_width_of_children.insert(root_id);
    }
    if root_size.height != layout_result.root_size.height {
        let root_id = layout_result.styled_dom.root.into_crate_internal().unwrap();
        parents_that_need_to_recalc_height_of_children.insert(root_id);
    }

    // parents need to be adjust before children
    for ParentWithNodeDepth { depth: _, node_id } in layout_result.styled_dom.non_leaf_nodes.iter() {

        let node_id = match node_id.into_crate_internal() { Some(s) => s, None => continue, };
        let parent_id = layout_result.styled_dom.node_hierarchy.as_container()[node_id].parent_id().unwrap_or(layout_result.styled_dom.root.into_crate_internal().unwrap());
        let node_data = &layout_result.styled_dom.node_data.as_container()[node_id];

        let changes_for_this_node = match nodes_to_relayout.get(&node_id) {
            None => continue,
            Some(s) => s,
        };

        let mut preferred_width_changed = (false, 0.0);
        let mut preferred_height_changed = (false, 0.0);
        let mut padding_x_changed = false;
        let mut padding_y_changed = false;
        let mut margin_x_changed = false;
        let mut margin_y_changed = false;

        let solved_width_layout = &mut layout_result.width_calculated_rects.as_ref_mut()[node_id];
        let solved_height_layout = &mut layout_result.height_calculated_rects.as_ref_mut()[node_id];
        let css_property_cache = layout_result.styled_dom.get_css_property_cache();

        if changes_for_this_node.contains_key(&CssPropertyType::MinWidth) ||
           changes_for_this_node.contains_key(&CssPropertyType::MaxWidth) ||
           changes_for_this_node.contains_key(&CssPropertyType::Width) {
            let styled_node_state = &layout_result.styled_dom.styled_nodes.as_container()[node_id].state;
            let wh_config = WhConfig {
                width: WidthConfig {
                    exact: css_property_cache.get_width(node_data, &node_id, styled_node_state).and_then(|p| p.get_property().copied()),
                    max: css_property_cache.get_max_width(node_data, &node_id, styled_node_state).and_then(|p| p.get_property().copied()),
                    min: css_property_cache.get_min_width(node_data, &node_id, styled_node_state).and_then(|p| p.get_property().copied()),
                },
                height: HeightConfig::default(),
            };

            let parent_width = layout_result.preferred_widths.as_ref()[parent_id].clone().unwrap_or(root_size.width as f32);
            let new_preferred_width = determine_preferred_width(&wh_config, layout_result.preferred_widths.as_ref()[node_id], parent_width);
            if new_preferred_width != solved_width_layout.preferred_width {
                preferred_width_changed = (true, solved_width_layout.preferred_width.min_needed_space().unwrap_or(0.0));
                solved_width_layout.preferred_width = new_preferred_width;
            }
        }

        if changes_for_this_node.contains_key(&CssPropertyType::MinHeight) ||
           changes_for_this_node.contains_key(&CssPropertyType::MaxHeight) ||
           changes_for_this_node.contains_key(&CssPropertyType::Height) {
            let styled_node_state = &layout_result.styled_dom.styled_nodes.as_container()[node_id].state;
            let wh_config = WhConfig {
                width: WidthConfig::default(),
                height: HeightConfig {
                    exact: css_property_cache.get_height(node_data, &node_id, &styled_node_state).and_then(|p| p.get_property().copied()),
                    max: css_property_cache.get_max_height(node_data, &node_id, &styled_node_state).and_then(|p| p.get_property().copied()),
                    min: css_property_cache.get_min_height(node_data, &node_id, &styled_node_state).and_then(|p| p.get_property().copied()),
                },
            };
            let parent_height = layout_result.preferred_heights.as_ref()[parent_id].clone().unwrap_or(root_size.height as f32);
            let new_preferred_height = determine_preferred_height(&wh_config, layout_result.preferred_heights.as_ref()[node_id], parent_height);
            if new_preferred_height != solved_height_layout.preferred_height {
                preferred_height_changed = (true, solved_height_layout.preferred_height.min_needed_space().unwrap_or(0.0));
                solved_height_layout.preferred_height = new_preferred_height;
            }
        }

        // width
        if let Some(prop) = changes_for_this_node.get(&CssPropertyType::PaddingLeft) {
            if let CssProperty::PaddingLeft(prop) = &prop.current_prop {
                solved_width_layout.padding_left = Some(*prop);
            } else {
                solved_width_layout.padding_left = None;
            }
            padding_x_changed = true;
        }

        if let Some(prop) = changes_for_this_node.get(&CssPropertyType::PaddingRight) {
            if let CssProperty::PaddingRight(prop) = &prop.current_prop {
                solved_width_layout.padding_right = Some(*prop);
            } else {
                solved_width_layout.padding_right = None;
            }
            padding_x_changed = true;
        }

        if let Some(prop) = changes_for_this_node.get(&CssPropertyType::MarginLeft) {
            if let CssProperty::MarginLeft(prop) = &prop.current_prop {
                solved_width_layout.margin_left = Some(*prop);
            } else {
                solved_width_layout.margin_left = None;
            }
            margin_x_changed = true;
        }

        if let Some(prop) = changes_for_this_node.get(&CssPropertyType::MarginRight) {
            if let CssProperty::MarginRight(prop) = &prop.current_prop {
                solved_width_layout.margin_right = Some(*prop);
            } else {
                solved_width_layout.margin_right = None;
            }
            margin_x_changed = true;
        }

        // height
        if let Some(prop) = changes_for_this_node.get(&CssPropertyType::PaddingTop) {
            if let CssProperty::PaddingTop(prop) = &prop.current_prop {
                solved_height_layout.padding_top = Some(*prop);
            } else {
                solved_height_layout.padding_top = None;
            }
            padding_y_changed = true;
        }

        if let Some(prop) = changes_for_this_node.get(&CssPropertyType::PaddingBottom) {
            if let CssProperty::PaddingBottom(prop) = &prop.current_prop {
                solved_height_layout.padding_bottom = Some(*prop);
            } else {
                solved_height_layout.padding_bottom = None;
            }
            padding_y_changed = true;
        }

        if let Some(prop) = changes_for_this_node.get(&CssPropertyType::MarginTop) {
            if let CssProperty::MarginTop(prop) = &prop.current_prop {
                solved_height_layout.margin_top = Some(*prop);
            } else {
                solved_height_layout.margin_top = None;
            }
            margin_y_changed = true;
        }

        if let Some(prop) = changes_for_this_node.get(&CssPropertyType::MarginBottom) {
            if let CssProperty::MarginBottom(prop) = &prop.current_prop {
                solved_height_layout.margin_bottom = Some(*prop);
            } else {
                solved_height_layout.margin_bottom = None;
            }
            margin_y_changed = true;
        }

        if preferred_width_changed.0 {
            // need to recalc the width of the node
            // need to bubble the width to the parent width
            // need to recalc the width of all children
            // need to recalc the x position of all siblings
            parents_that_need_to_recalc_width_of_children.insert(parent_id);
            nodes_that_need_to_bubble_width.insert(node_id, preferred_width_changed.1);
            parents_that_need_to_recalc_width_of_children.insert(node_id);
            parents_that_need_to_reposition_children_x.insert(parent_id);
        }

        if preferred_height_changed.0 {
            // need to recalc the height of the node
            // need to bubble the height of all current node siblings to the parent height
            // need to recalc the height of all children
            // need to recalc the y position of all siblings
            parents_that_need_to_recalc_height_of_children.insert(parent_id);
            nodes_that_need_to_bubble_height.insert(node_id, preferred_height_changed.1);
            parents_that_need_to_recalc_height_of_children.insert(node_id);
            parents_that_need_to_reposition_children_y.insert(parent_id);
        }

        if padding_x_changed {
            // need to recalc the widths of all children
            // need to recalc the x position of all children
            parents_that_need_to_recalc_width_of_children.insert(node_id);
            parents_that_need_to_reposition_children_x.insert(node_id);
        }

        if padding_y_changed {
            // need to recalc the heights of all children
            // need to bubble the height of all current node children to the current node min_inner_size_px
            parents_that_need_to_recalc_height_of_children.insert(node_id);
            parents_that_need_to_reposition_children_y.insert(node_id);
        }

        if margin_x_changed {
            // need to recalc the widths of all siblings
            // need to recalc the x positions of all siblings
            parents_that_need_to_recalc_width_of_children.insert(parent_id);
            parents_that_need_to_reposition_children_x.insert(parent_id);
        }

        if margin_y_changed {
            // need to recalc the heights of all siblings
            // need to recalc the y positions of all siblings
            parents_that_need_to_recalc_height_of_children.insert(parent_id);
            parents_that_need_to_reposition_children_y.insert(parent_id);
        }

        /*
        let layout_position = layout_result.layout_positions.as_ref()[node_id];
        // TODO: recalc absolute positions!
        let layout_position_changed = nodes_that_have_changed_layout_position.contains(&node_id);

        if layout_position_changed && layout_position != LayoutPosition::Static {
            if changes_for_this_node.get(&CssPropertyType::Top).is_some() ||
               changes_for_this_node.get(&CssPropertyType::Left).is_some() ||
               changes_for_this_node.get(&CssPropertyType::Right).is_some() ||
               changes_for_this_node.get(&CssPropertyType::Bottom).is_some() {
                // TODO: recalc the absolute position for this node
                // if the width / height / position changed, insert accordingly
                // let new_absolute_width = determine_child_width_absolute();
                // let new_absolute_height = determine_child_height_absolute();
                // let new_absolute_x = determine_child_x_absolute();
                // let new_absolute_y = determine_child_y_absolute();
            }
        }
        */
    }

    // for all nodes that changed, recalculate the min_inner_size_px of the parents
    // by re-bubbling the sizes to the parents (but only for the nodes that need it)
    let mut rebubble_parent_widths = BTreeMap::new();
    let mut rebubble_parent_heights = BTreeMap::new();

    for (node_id, old_preferred_width) in nodes_that_need_to_bubble_width.iter() {
        if let Some(parent_id) = layout_result.styled_dom.node_hierarchy.as_container()[*node_id].parent_id() {
            let new_preferred_width = layout_result.width_calculated_rects.as_ref()[*node_id].preferred_width;
            let change = new_preferred_width.min_needed_space().unwrap_or(0.0) - old_preferred_width;
            layout_result.width_calculated_rects.as_ref_mut()[parent_id].min_inner_size_px += change;
            if change != 0.0 {
                *rebubble_parent_widths.entry(parent_id).or_insert_with(|| 0.0) += change;
                parents_that_need_to_recalc_width_of_children.insert(parent_id);
            }
        }
    }

    // recalc(&mut layout_result.preferred_heights);

    for (node_id, old_preferred_height) in nodes_that_need_to_bubble_height.iter() {
        if let Some(parent_id) = layout_result.styled_dom.node_hierarchy.as_container()[*node_id].parent_id() {
            let new_preferred_height = layout_result.height_calculated_rects.as_ref()[*node_id].preferred_height;
            let change = new_preferred_height.min_needed_space().unwrap_or(0.0) - old_preferred_height;
            layout_result.height_calculated_rects.as_ref_mut()[parent_id].min_inner_size_px += change;
            if change != 0.0 {
                *rebubble_parent_heights.entry(parent_id).or_insert_with(|| 0.0) += change;
                parents_that_need_to_recalc_height_of_children.insert(parent_id);
            }
        }
    }

    // propagate min_inner_size_px change from the inside out
    for ParentWithNodeDepth { depth: _, node_id } in layout_result.styled_dom.non_leaf_nodes.iter().rev() {

        let node_id = match node_id.into_crate_internal() { Some(s) => s, None => continue, };

        if let Some(change_amount) = rebubble_parent_widths.remove(&node_id) {
            layout_result.width_calculated_rects.as_ref_mut()[node_id].min_inner_size_px += change_amount;
            if let Some(parent_id) = layout_result.styled_dom.node_hierarchy.as_container()[node_id].parent_id() {
                *rebubble_parent_widths.entry(parent_id).or_insert_with(|| 0.0) += change_amount;
                parents_that_need_to_recalc_width_of_children.insert(parent_id);
            }
        }

        if let Some(change_amount) = rebubble_parent_heights.remove(&node_id) {
            layout_result.height_calculated_rects.as_ref_mut()[node_id].min_inner_size_px += change_amount;
            if let Some(parent_id) = layout_result.styled_dom.node_hierarchy.as_container()[node_id].parent_id() {
                *rebubble_parent_heights.entry(parent_id).or_insert_with(|| 0.0) += change_amount;
                parents_that_need_to_recalc_height_of_children.insert(parent_id);
            }
        }
    }

    // now for all nodes that need to recalculate their width, calculate their flex_grow_px,
    // then recalculate the width of their children, but STOP recalculating once a child
    // with an exact width is found
    width_calculated_rect_arena_apply_flex_grow(
        &mut layout_result.width_calculated_rects.as_ref_mut(),
        &layout_result.styled_dom.node_hierarchy.as_container(),
        &layout_result.layout_flex_grows.as_ref(),
        &layout_result.layout_positions.as_ref(),
        &layout_result.layout_flex_directions.as_ref(),
        &layout_result.styled_dom.non_leaf_nodes.as_ref(),
        root_size.width as f32,
        &parents_that_need_to_recalc_width_of_children // <- important - only recalc the widths necessary!
    );

    height_calculated_rect_arena_apply_flex_grow(
        &mut layout_result.height_calculated_rects.as_ref_mut(),
        &layout_result.styled_dom.node_hierarchy.as_container(),
        &layout_result.layout_flex_grows.as_ref(),
        &layout_result.layout_positions.as_ref(),
        &layout_result.layout_flex_directions.as_ref(),
        &layout_result.styled_dom.non_leaf_nodes.as_ref(),
        root_size.height as f32,
        &parents_that_need_to_recalc_height_of_children // <- important - only recalc the heights necessary!
    );

    // -- step 2: recalc position for those parents that need it

    get_x_positions(
        &mut layout_result.solved_pos_x.as_ref_mut(),
        &layout_result.width_calculated_rects.as_ref(),
        &layout_result.styled_dom.node_hierarchy.as_container(),
        &layout_result.layout_positions.as_ref(),
        &layout_result.layout_flex_directions.as_ref(),
        &layout_result.layout_justify_contents.as_ref(),
        &layout_result.styled_dom.non_leaf_nodes.as_ref(),
        LogicalPosition::new(root_bounds.origin.x as f32, root_bounds.origin.y as f32),
        &parents_that_need_to_reposition_children_x, // <- important
    );

    get_y_positions(
        &mut layout_result.solved_pos_y.as_ref_mut(),
        &layout_result.height_calculated_rects.as_ref(),
        &layout_result.styled_dom.node_hierarchy.as_container(),
        &layout_result.layout_positions.as_ref(),
        &layout_result.layout_flex_directions.as_ref(),
        &layout_result.layout_justify_contents.as_ref(),
        &layout_result.styled_dom.non_leaf_nodes.as_ref(),
        LogicalPosition::new(root_bounds.origin.x as f32, root_bounds.origin.y as f32),
        &parents_that_need_to_reposition_children_y, // <- important
    );

    // update positioned_word_cache
    let mut updated_word_caches = parents_that_need_to_recalc_width_of_children.clone();
    for parent_id in parents_that_need_to_recalc_width_of_children.iter() {
        for child_id in parent_id.az_children(&layout_result.styled_dom.node_hierarchy.as_container()) {
            // if max_width_changed { } // - optimization?
            updated_word_caches.insert(child_id);
        }
    }

    #[cfg(feature = "text_layout")]
    create_word_positions(
        &mut layout_result.positioned_words_cache,
        &updated_word_caches,
        &pipeline_id,
        app_resources,
        &layout_result.words_cache,
        &layout_result.shaped_words_cache,
        &layout_result.styled_dom,
        &layout_result.width_calculated_rects.as_ref(),
    );

    // determine which nodes changed their size and return
    let mut nodes_that_changed_size = BTreeSet::new();
    for parent_id in parents_that_need_to_recalc_width_of_children {
        nodes_that_changed_size.insert(parent_id);
        for child_id in parent_id.az_children(&layout_result.styled_dom.node_hierarchy.as_container()) {
            nodes_that_changed_size.insert(child_id);
        }
    }
    for parent_id in parents_that_need_to_recalc_height_of_children {
        nodes_that_changed_size.insert(parent_id);
        for child_id in parent_id.az_children(&layout_result.styled_dom.node_hierarchy.as_container()) {
            nodes_that_changed_size.insert(child_id);
        }
    }

    let css_property_cache = layout_result.styled_dom.get_css_property_cache();
    let node_data_container = layout_result.styled_dom.node_data.as_container();

    let mut all_offsets_to_recalc = BTreeMap::new();
    for node_id in nodes_that_changed_size.iter() {

        all_offsets_to_recalc.entry(*node_id).or_insert_with(|| {
            let styled_node_state = &layout_result.styled_dom.styled_nodes.as_container()[*node_id].state;
            precalculate_offset(&node_data_container[*node_id], &css_property_cache, node_id, styled_node_state)
        });

        for child_id in node_id.az_children(&layout_result.styled_dom.node_hierarchy.as_container()) {
            all_offsets_to_recalc.entry(child_id).or_insert_with(|| {
                let styled_node_state = &layout_result.styled_dom.styled_nodes.as_container()[child_id].state;
                precalculate_offset(&node_data_container[*node_id], &css_property_cache, &child_id, styled_node_state)
            });
        }
    }

    // update layout_result.rects and layout_result.glyph_cache
    // if positioned_word_cache changed, regenerate layouted_glyph_cache
    position_nodes(
        &mut layout_result.rects.as_ref_mut(),
        &layout_result.styled_dom,
        AllOffsetsProvider::OnlyRecalculatedNodes(&all_offsets_to_recalc),
        &layout_result.width_calculated_rects.as_ref(),
        &layout_result.height_calculated_rects.as_ref(),
        &layout_result.solved_pos_x.as_ref(),
        &layout_result.solved_pos_y.as_ref(),
        &nodes_that_changed_size,
        &nodes_that_changed_size,
        &layout_result.layout_positions.as_ref(),
        &layout_result.words_cache,
        &layout_result.shaped_words_cache,
        &layout_result.positioned_words_cache,
        pipeline_id,
    );

    layout_result.root_size = root_bounds.size;
    layout_result.root_position = root_bounds.origin;

    // TODO: optimize?
    get_nodes_that_need_scroll_clip(
        &mut layout_result.scrollable_nodes,
        &layout_result.styled_dom.styled_nodes.as_container(),
        &layout_result.styled_dom.node_data.as_container(),
        &layout_result.rects.as_ref(),
        &layout_result.styled_dom.non_leaf_nodes.as_ref(),
        pipeline_id,
    );

    nodes_that_changed_size.into_iter().collect()
}
