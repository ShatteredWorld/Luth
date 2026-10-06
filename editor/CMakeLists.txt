add_library(LuthienLib STATIC

    source/lepch.cpp

    source/luthien/CommandHistory.cpp
    source/luthien/Editor.cpp
    source/luthien/EditorAutoSave.cpp
    source/luthien/EditorCamera.cpp
    source/luthien/EditorClipboard.cpp
    source/luthien/EditorHooks.cpp
    source/luthien/EditorSelection.cpp
    source/luthien/EditorSettings.cpp
    source/luthien/EditorStyle.cpp
    source/luthien/PlayModeController.cpp
    source/luthien/ProjectLauncher.cpp
    source/luthien/SceneViewStore.cpp
    source/luthien/Workspace.cpp

    source/luthien/commands/AssetCommands.cpp
    source/luthien/commands/EntityCommands.cpp

    source/luthien/inspectors/ComponentDrawerRegistry.cpp
    source/luthien/inspectors/FontViewer.cpp
    source/luthien/inspectors/MaterialEditor.cpp
    source/luthien/inspectors/ModelViewer.cpp
    source/luthien/inspectors/PhysicsMaterialEditor.cpp
    source/luthien/inspectors/SceneViewer.cpp
    source/luthien/inspectors/ShaderEditor.cpp
    source/luthien/inspectors/TextureEditor.cpp

    source/luthien/inspectors/component_drawers/AnimationControllerDrawer.cpp
    source/luthien/inspectors/component_drawers/AnimationDrawer.cpp
    source/luthien/inspectors/component_drawers/BoneAttachmentDrawer.cpp
    source/luthien/inspectors/component_drawers/CameraDrawer.cpp
    source/luthien/inspectors/component_drawers/CharacterControllerDrawer.cpp
    source/luthien/inspectors/component_drawers/ColliderDrawer.cpp
    source/luthien/inspectors/component_drawers/DebugDrawers.cpp
    source/luthien/inspectors/component_drawers/DirectionalLightDrawer.cpp
    source/luthien/inspectors/component_drawers/FogVolumeDrawer.cpp
    source/luthien/inspectors/component_drawers/MeshRendererDrawer.cpp
    source/luthien/inspectors/component_drawers/PointLightDrawer.cpp
    source/luthien/inspectors/component_drawers/RegisterComponentDrawers.cpp
    source/luthien/inspectors/component_drawers/RigidBodyDrawer.cpp
    source/luthien/inspectors/component_drawers/SpotLightDrawer.cpp
    source/luthien/inspectors/component_drawers/TransformDrawer.cpp
    source/luthien/inspectors/component_drawers/WindDrawer.cpp

    source/luthien/panels/ConsolePanel.cpp
    source/luthien/panels/EditorSettingsWindow.cpp
    source/luthien/panels/FrameDebuggerPanel.cpp
    source/luthien/panels/GamePanel.cpp
    source/luthien/panels/HierarchyPanel.cpp
    source/luthien/panels/HistoryPanel.cpp
    source/luthien/panels/InspectorPanel.cpp
    source/luthien/panels/MaterialGraphPanel.cpp
    source/luthien/panels/ProfilerPanel.cpp
    source/luthien/panels/ProjectPanel.cpp
    source/luthien/panels/RenderPanel.cpp
    source/luthien/panels/ResourcePanel.cpp
    source/luthien/panels/ScenePanel.cpp
    source/luthien/panels/TextureRemapDialog.cpp

    source/luthien/viewport/GizmoController.cpp
    source/luthien/viewport/ViewportOverlays.cpp
    source/luthien/viewport/ViewportRenderer.cpp

    source/luthien/widgets/AssetSlot.cpp
    source/luthien/widgets/ButtonGroup.cpp
    source/luthien/widgets/CategoryPopup.cpp
    source/luthien/widgets/Charts.cpp
    source/luthien/widgets/CollapsingHeader.cpp
    source/luthien/widgets/FilterBox.cpp
    source/luthien/widgets/HelpMarker.cpp
    source/luthien/widgets/InfoTable.cpp
    source/luthien/widgets/InspectorHeader.cpp
    source/luthien/widgets/Properties.cpp
    source/luthien/widgets/Splitter.cpp
    source/luthien/widgets/TexturePreview.cpp
    source/luthien/widgets/ThumbnailCache.cpp
    source/luthien/widgets/ThumbnailGenerator.cpp
    source/luthien/widgets/ThumbnailPreviewScene.cpp
)
add_library(Luth::Luthien ALIAS LuthienLib)

target_compile_features(LuthienLib PUBLIC cxx_std_20)
target_precompile_headers(LuthienLib PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/source/lepch.h")
target_include_directories(LuthienLib
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/source"
)
target_link_libraries(LuthienLib
    PUBLIC
        Luth::Luth
        ImGui::ImGui
        ImGuizmo::ImGuizmo
)

luth_enable_tracy_fibers(LuthienLib)
luth_enable_asan(LuthienLib)
luth_set_output_directories(LuthienLib)
set_property(TARGET LuthienLib PROPERTY FOLDER "Luthien")
