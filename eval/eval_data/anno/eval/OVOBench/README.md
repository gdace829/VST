---
license: mit
extra_gated_prompt: >-
  You agree to not use the dataset to conduct experiments that cause harm to
  human subjects. Please note that the data in this dataset may be subject to
  other agreements. Before using the data, be sure to read the relevant
  agreements carefully to ensure compliant use. Video copyrights belong to the
  original video creators or platforms and are for academic research use only.
task_categories:
- visual-question-answering
- video-classification
extra_gated_fields:
  Name: text
  Company/Organization: text
  Country: text
  E-Mail: text
modalities:
- Video
- Text
configs:
- config_name: backward_tracking
  data_files: json/backward_tracking.json
- config_name: real_time_visual_perception
  data_files: json/real_time_visual_perception.json
- config_name: forward_active_responding
  data_files: json/forward_active_responding.json

language:
- en
size_categories:
- 1K<n<10K
---