python src/generate_pid_summary_and_embeddings.py \
  --data_path data/pid_panel.csv \
  --meta_path data/meta_information.csv \
  --theme_path data/theme_tags.csv \
  --scenario_path data/scenario_rules.json \
  --output_text_path outputs/pid_texts.csv \
  --output_embed_path outputs/pid_embeddings.npy
