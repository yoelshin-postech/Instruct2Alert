# Benchmark data

## Annotations (included)

The annotations in this folder (`datasets/safety_vlm_dataset.xlsx`, `video_sources.csv`) are licensed under [CC BY-NC 4.0](LICENSE). This license does not cover the videos.

`datasets/safety_vlm_dataset.xlsx`: 100 video–query scenarios over 85 videos, read directly by the pipeline.

| Column | Meaning |
|---|---|
| `Scenario_ID` | 1–100 |
| `Video_name` | file stem of the video in `input_video/` (`<Video_name>.mp4`) |
| `Source` | source dataset (see below) |
| `Length(sec)` | clip length |
| `Domain` | Industrial (28) / Public (55) / Traffic (17) |
| `Query_Type` | Event (50, instantaneous) / State (50, duration-conditioned) |
| `Scenario_Category` | instruction family label |
| `Duration (Sec)` | duration condition parsed from the query (State only) |
| `Primary_Target`, `Target` | main and contextual targets |
| `Scenario_EN` | the natural-language instruction given to the system |
| `GT_intervals_frame` | ground-truth alert intervals, native frame indices, 1-based inclusive |

Annotation protocol: an interval starts when the queried condition becomes observable and ends when it stops holding. For State queries the onset is shifted by the query duration (shorter occurrences are negative). No clearing tolerance is applied to ground truth. See Sec. 1 of the supplement.

## Videos (not included)

The videos are **not** redistributed. Obtain unedited videos from their original providers under each provider's terms, and place every video in `input_video/` as `<Video_name>.mp4`. `video_sources.csv` lists every file with its source, the scenarios that use it, and the original URL where one exists.

| `Source` value | Dataset | Videos | Scenarios | Where to obtain | Terms |
|---|---|---|---|---|---|
| `UCF` | UCF-Crime (Sultani et al., CVPR 2018) | 21 | 28 | [project page](https://www.crcv.ucf.edu/projects/real-world/); original file names, e.g. `Abuse001_x264` | research use only; cite the authors |
| `AI_hub` | AI-Hub CCTV datasets 171 / 71850 / 71953 (NIA, Korea) | 24 | 26 | [aihub.or.kr](https://aihub.or.kr) (application required) | AI-Hub terms; redistribution not permitted |
| `Youtube` | public YouTube videos | 15 | 17 | URLs in `video_sources.csv` (`v_007`–`v_010` are four clips of one video) | copyright of the uploaders |
| `Safety_Bench` | iSafetyBench (Abdullah et al., ICCVW 2025) | 9 | 13 | [project page](https://isafetybench.github.io); file names `<YouTube id>_trim_<n>` | per the iSafetyBench release |
| `NWPU` | NWPU Campus (Cao et al., CVPR 2023) | 9 | 9 | [project page](https://campusvad.github.io); e.g. `D035_01` | per the NWPU Campus release |
| `WEDA-FALL` | WEDA-FALL (Marques, 2023) | 7 | 7 | [GitHub](https://github.com/joaojtmarques/WEDA-FALL); `F01`–`F07` | per the WEDA-FALL release |

Some clips were trimmed or edited from their source videos for the benchmark, so the original source alone does not reproduce them. To request these clips for non-commercial research use, email **yoelshin@postech.ac.kr** with your name, affiliation, and intended use. Clips are provided under the terms of their original sources and must not be redistributed.

## Run outputs

`main_runs/` is filled by running the pipeline (see the main README).
