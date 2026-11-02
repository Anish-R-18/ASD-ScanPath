FINAL RESULTS PACKAGE - TPP-Gaze ASD/TD
Summary of the completed Phase 1, Phase 2, Saliency4ASD and corrected Gaze4ASD experiments.
No model was retrained or modified to build this package.

Current text:
  MAIN_FINDINGS.txt   current findings: in-domain likelihood, external likelihood, real ASD vs TD,
                      generated behaviour, temperature
  LIMITATIONS.txt     documented limitations; no clinical-validity claims
  SUPERSEDED.txt      which items below use the old Gaze4ASD mapping

Tables:
  1_main_results_saliency4asd_test.csv  Phase 1 vs Phase 2 (held-out test NLL + improvement)   VALID
  2_external_gaze4asd_table.csv         Old Gaze4ASD NLL + behavioural means                   SUPERSEDED
  3_participant_statistics.csv          Old Gaze4ASD effect sizes and tests                    SUPERSEDED

Figures (figures/):
  fig1_phase1_vs_phase2_nll.png                     Saliency4ASD likelihood                    VALID
  fig2_real_vs_gen_ASD_TD_distributions.png         old Gaze4ASD distributions                 SUPERSEDED
  fig3_effect_size_forest.png                       old Gaze4ASD effect sizes                  SUPERSEDED
  fig4a_representative_scanpaths_saliency4asd.png   Saliency4ASD examples, temperature 2.0     VALID
  fig4b_representative_scanpaths_gaze4asd.png       old Gaze4ASD examples                      SUPERSEDED
  fig5_real_vs_generated_similarity.png             old Gaze4ASD MultiMatch                    SUPERSEDED

Corrected Gaze4ASD results:
  results/external_gaze4asd_v2/         tracked summaries, tables and figures (start with README.md)
  training/runs/external_gaze4asd_v2/   full local outputs, not tracked by Git
