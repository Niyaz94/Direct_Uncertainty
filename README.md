# Direct Uncertainty With U-Net

Direct Uncertainty is a novel framework for modeling uncertainty in medical image segmentation. Unlike traditional ensemble or dropout-based methods, DU introduces a dedicated uncertainty class during training to explicitly capture regions where expert annotators disagree. This allows the model to learn uncertainty directly from data without requiring any changes to the underlying segmentation architecture (e.g., nnU-Net), and without increasing computational cost.

<pre> project-root/ 
├── c_unet/ # Modified nnU-Net source code (custom U-Net implementation) 
├── model_bash/ # Bash scripts for training and prediction workflows 
├── model_configs/ # JSON configuration files for nnU-Net experiments 
├── model_logs/ # Training logs, metrics, and monitoring output 
├── model_predictions/ # Output predictions from trained models 
├── results/ # Final results (aggregated, visualized, or post-processed) 
├── README.md # Project description and usage guide (this file) 
</pre>