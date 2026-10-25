# Benchmark data

## Annotations (included)

The annotations in this folder (`datasets/safety_vlm_dataset.xlsx`, `video_sources.csv`) are licensed under [CC BY-NC 4.0](LICENSE). This license does not cover the videos.

`datasets/safety_vlm_dataset.xlsx`: 100 video–query scenarios over 85 videos, read directly by the pipeline.

| Column | Meaning |
|---|---|
| `Scenario_ID` | 1–100 |
| `Video_name` | file stem of the video in `input_video/` (`<Video_name>.mp4`) |
| `Source` | source dataset (see below) |
| `Length(sec)` | approximate clip length in whole seconds |
| `Domain` | Industrial (28) / Public (55) / Traffic (17) |
| `Query_Type` | Event (50, instantaneous) / State (50, duration-conditioned) |
| `Scenario_Category` | scenario category, one of 13 labels such as Lying, Loitering, Security, Fire |
| `Duration (Sec)` | duration condition parsed from the query (State only) |
| `Primary_Target`, `Target` | main and contextual targets |
| `Scenario_EN` | the natural-language instruction given to the system |
| `GT_intervals_frame` | ground-truth alert intervals as native frame indices, 1-based inclusive; evaluation converts `[start, end]` to seconds as `[(start - 1) / fps, end / fps]` with the frame rate the decoder reports |

Annotation protocol: an interval starts when the queried condition becomes observable and ends when it stops holding. For State queries the onset is shifted by the query duration (shorter occurrences are negative). No clearing tolerance is applied to ground truth. See Sec. 1 of the supplement.

## Videos (not included)

The videos are **not** redistributed. Obtain unedited videos from their original providers under each provider's terms, and place every video in `input_video/` as `<Video_name>.mp4`.

| `Source` value | Dataset | Videos | Scenarios | Where to obtain | Terms |
|---|---|---|---|---|---|
| `UCF` | UCF-Crime (Sultani et al., CVPR 2018) | 21 | 28 | [project page](https://www.crcv.ucf.edu/projects/real-world/); original file names, e.g. `Abuse001_x264` | research use only; cite the authors |
| `AI_hub` | AI-Hub CCTV datasets 171 / 71850 / 71953 (NIA, Korea) | 24 | 26 | [aihub.or.kr](https://aihub.or.kr) (application required) | AI-Hub terms; redistribution not permitted |
| `Youtube` | public YouTube videos | 15 | 17 | URLs and the scenarios using each clip in `video_sources.csv` (`v_007`–`v_010` are four clips of one video) | copyright of the uploaders |
| `Safety_Bench` | iSafetyBench (Abdullah et al., ICCVW 2025) | 9 | 13 | [project page](https://isafetybench.github.io); file names `<YouTube id>_trim_<n>` | per the iSafetyBench release |
| `NWPU` | NWPU Campus (Cao et al., CVPR 2023) | 9 | 9 | [project page](https://campusvad.github.io); e.g. `D035_01` | academic, non-commercial research only; redistribution not permitted |
| `WEDA-FALL` | WEDA-FALL (Marques, 2023) | 7 | 7 | [GitHub](https://github.com/joaojtmarques/WEDA-FALL); `F01`–`F07` | per the WEDA-FALL release |

Some clips were trimmed or edited from their source videos for the benchmark, so the original files alone do not match them. The clips themselves are not shared. To recreate them from the original videos, request the editing details (the source video and the start and end time of each clip) from **yoelshin@postech.ac.kr**.

## Run outputs

`main_runs/` is filled by running the pipeline (see the main README).
