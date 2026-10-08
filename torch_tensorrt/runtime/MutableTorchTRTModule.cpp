--- a/torch_tensorrt/runtime/MutableTorchTRTModule.cpp
+++ b/torch_tensorrt/runtime/MutableTorchTRTModule.cpp
@@ -10,8 +10,17 @@ void MutableTorchTRTModule::update_refit_condition() {
-    if (engine_->needsRefit()) {
-        refit_flag_ = RefitFlag::NEEDS_REFIT;
-    } else {
-        refit_flag_ = RefitFlag::LIVE;
+    // Three-step validation for CUDA 13.x on B100/H100
+    // Step 1: Check if any weight-affecting attributes changed
+    bool weight_affecting_change = check_weight_affecting_attribute_change();
+    // Step 2: If no weight-affecting changes, directly set to LIVE
+    if (!weight_affecting_change) {
+        refit_flag_ = RefitFlag::LIVE;
+        return;
     }
+    // Step 3: Otherwise, delegate to TensorRT's needsRefit
+    if (engine_->needsRefit()) {
+        refit_flag_ = RefitFlag::NEEDS_REFIT;
+    } else {
+        refit_flag_ = RefitFlag::LIVE;
+    }
 }