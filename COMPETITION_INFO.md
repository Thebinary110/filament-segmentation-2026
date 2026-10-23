# Solar Filament Segmentation Challenge 2026

Source: https://www.kaggle.com/competitions/filament-segmentation-2026 (fetched 2026-10-14 via Kaggle API; rules excluded)

## Metadata (Kaggle API)
- Title: Solar Filament Segmentation Challenge 2026
- Subtitle/description: Automatic Segmentation of Solar Filaments using Computer Vision algorithms (with or without fitting) from GONG's H-Alpha Observations
- Host: Earth-Space AI Research (ESAIR) Lab
- Category: Community
- Reward: 3,000 USD
- Evaluation metric: panoptic_quality (custom metric; formula is on the Evaluation tab, not available via API)
- Tags: custom metric
- License: CC BY-NC 4.0
- Competition id: 143678
- Created: 2026-05-20; enabled: 2026-07-08
- Deadline / merger deadline: 2026-11-15 06:00 UTC
- Max team size: 10; max daily submissions: 5
- Teams: 1008 (at fetch time); awards points: no; code-only submissions: no
- Top public leaderboard score at fetch time: 0.63

## Data (from files + annotation JSON)
- Root: MAGFiLO_1.0_Kaggle_2026/
  - train/train_images (707 jpeg), train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
  - test/test_images (180 jpeg), no annotations
- Total download: ~671 MB zip, 888 files
- Format: COCO-style JSON (info, licenses, categories, images, annotations)
- Info: "MLEcoFi 2024 Dataset - V1.0", https://www.mlecofi.net/, version 1.0, contributor MLEcoFi Team, created 2024-07-15, updated 2026-06-01, team Earth-Space AI Research Lab (www.esairlab.com), collaborator National Solar Observatory / Georgia State University
- License entry: NSO/GONG (https://gong.nso.edu/)
- Categories (supercategory "filament"): 1 Left, 2 Right, 3 Unidentifiable, 4 Ambiguous
- Images: 2048x2048 GONG H-alpha; entries have license, file_name, url (gong2.nso.edu), height, width, date_captured, id
- Annotations: 8199 total (cat 1: 2535, cat 2: 2590, cat 3: 3074, cat 4: none in train); fields: segmentation (polygon), area, iscrowd, spine (polyline), image_id, bbox, category_id, id (uuid)
- Note: `images` has 1154 entries but only 707 unique file_names; ids look like "<prefix>-<filename stem>", so the same file appears under several ids.

## Full page text (via Kaggle pages API; rules pages omitted)

## Page: data-description
For this competition, you will need (1) images and (2) ground-truth annotations. Both of these items can be downloaded in this page. Below, click on the "Download All" button, and then choose "Download as zip". When the download is complete, unzip the downloaded file. The content of the unzipped file is shown below:

```text
MAGFiLO_1.0_Kaggle_2026/
|-- train/
|   |-- train_images/
|   |-- MAGFiLO_1.0_Annotations_kaggle2026_train.json
|-- test/
    |-- test_images/
```

### Images and Annotations
The goal is to identify solar filaments from solar images. Solar filaments are best visible in H-Alpha images. Therefore, we use the observations (images) captured by H-Alpha instruments of Global Oscillations Network Group ([GONG](https://en.wikipedia.org/wiki/Global_Oscillations_Network_Group)). These are the same images used for creating the manual annotation of solar filaments available in MAGFiLO.

These H-Alpha observations are images of size `2048 x 2048` pixels, which, for the sake of this competition, have been converted from their original FITS format to the standard 8-bit JPEG format. Contestants should note that these images are grayscale and should not be processed as color images (e.g., RGB images).

Each observation's capture time and instrument are encoded in its file name as follows: `YYYYMMDDHHMMSSII`. For example, `20260901165702Bh.jpeg` corresponds to the capture time of September 1, 2026, at 16:57:02, using the Big Bear (Bh) Solar Observatory. 

The H-alpha observations in MAGFiLO often contain multiple filaments. Further, all filaments present within one image may be annotated multiple times by different annotators, independently. As an example, the image with `file_name` as `20260901165702Bh.jpeg` may have two sets of annotations, e.g., `010101-20260901165702Bh` and `010102-20260901165702Bh` (note that they are different by only one digit). You may treat those images simply as "different" images.

For this competition, the ground-truth data are the filament **segmentations**. For storage efficiency, instead of binary masks, we store segmentations as polygons in [RLE](https://en.wikipedia.org/wiki/Run-length_encoding) format. The conversion (RLE <-> Binary Masks) are lossless. 

The annotations are structured into the [COCO-style data format](https://cocodataset.org/#format-data). That means, you can take advantage of many common operations using the `pycocotools` python package (see [https://github.com/ppwwyyxx/cocoapi](https://github.com/ppwwyyxx/cocoapi)). 

### More Info about JSON File

```json
{
"info": info,
"images": [image],
"annotations": [annotation],
"licenses": [license],
"categories": [category],
}

info{
"year": int,
"version": str,
"description": str,
"contributor": str,
"url": str,
"date_created": datetime,
}

license{
"id": int,
"name": str,
"url": str,
}

categories[
{
   "id": int,
   "name": str,
   "supercategory": str,
}]

annotation{
"id": str,
"image_id": int,
"category_id": int,
"segmentation": [[]],
"area": float,
"spine": [],
"bbox": [x, y, width, height],
"iscrowd": 0 or 1,
}

image{
"id": str,
"width": int,
"height": int,
"file_name": str,
"license": int,
"date_captured": datetime,
}
```

- **info:** corresponds to a dictionary containing metadata description of the dataset.
- **images:** corresponds to a list of image dictionaries.
- **annotations:** corresponds to a list of annotation dictionaries.
- **licenses:** corresponds to a dictionary containing the license information of the GONG H-Alpha images.
- **categories:** corresponds to a list of category dictionaries. 
- **image:** corresponds to a dictionary containing information  about an annotated image, including its name and downloadable URL.
- **annotation:** corresponds to a dictionary containing information about an annotated filament, including its segmentation and bounding box.
- **license:** corresponds to a dictionary describing the license of an image.
- **category:** corresponds to a dictionary containing the category information that each filament may be described with. 

#### More Information on Annotations

- A bounding box in `annotation["bbox"]` is represented with a list of 4 values,  `[x, y, width, height]` , where `x,y` corresponds to the top-left corner of the box. All values are in pixels unit.
- A segmentation in `annotation["segmentation"]` is a list of floats representing a closed path. Each path forms a polygon capturing a filament's shape. A polygon made with n points is represented as `[x_0, y_0, x_1, y_1, ..., x_(n-1), y_(n-1)]` where `x_0 = x_(n-1)` and `y_0 = y_(n-1)`.
- The area in  `annotation["area"]` is the segmentation area (not the bbox area). This is computed using pycocoapi package, by first converting the polygon into RLE.
- A spine in `annotation["spine"]` is a list of floats representing a path. Each path captures a filament's spine. Disconnected line segments are not allowed. A spine with `n` points is represented as `[x_0, y_0, x_1, y_1, ..., x_(n-1), y_(n-1)]`.
- The tag `annotation["iscrowd"]` is always zero indicating that each segmentation corresponds to a single filaments (i.e., no group of filaments is annotated together).
- Although `annotation["segmentation"]` is a list, it only contains one polygon. That is, each filament is annotated using a single-piece polygon.
- The class labels listed in `category["name"]` are: "Left" (id = 1), "Right", (id = 2), "Unidentifiable" (id = 3), and "Ambiguous" (id = 4).
- The image ids in `image["id"]`  is a string that shows the annotator's batch name, as well as the  name (together, unique). For example, id `010401-20160920230134Lh` indicates that the image with name `20160920230134Lh.jpg` is annotated by the annotator `010401`, and it might have also been annotated by two other annotators, `010402` and `010403`, in the same group.
- The annotation ids in `annotation["id"]` is a string (e.g., `a7639f8a-c76d-43a7-b392-e75842262b75`) generated by the annotation platform, which is unique for filaments. 
## Page: Announcements
- **[Sep 20, 2026]** Reminder: The SABID workshop of IEEE BigData Conference is interested in the topic of this challenge. Please consider submitting your reports to this workshop: https://sites.google.com/view/sabid2026/home .   
- **[Aug 20, 2026]** Looking at some submissions, it is clear to us that some participants are spending their time and energy on hacking the system rather tackling the actual problem. We encourage honest participants to continue their work without comparing their scores with others. Any PQ score of greater than 0.35 is of great value to us.  
- **[Aug 12, 2026]** Our leaderboard is now re-scored. 
- **[Aug 9, 2026]** We added a self-evaluation notebook for participants. Please see the "Self Evaluation" section.
- **[Aug 7, 2026]** We are going to update our evaluation methodology to address the raised issues. The details are explained below. The new scoring metric will be in effect soon; all previous submissions will be updated as well. The exact time the leaderboard is to be updated depends on when Kaggle resets the scoring mechanism.
## Page: Important Dates
- July 10, 2026: Competition is launched. Contestants can start working on the challenge.
- Nov 15, 2026, 11:59 pm CDT: Deadline for contest teams to submit final reports and solutions.
- Nov 30, 2026: The winning teams will be announced.
- Dec 14-17, 2026: (Optional) IEEE BigData conference will take place at Phoenix, AZ. The winners of this BigDataCup competition (as well as all other BigDataCups) will be announced
## Page: abstract
*(See the Announcement section below for updates.)*

# TASK  
Participants are invited to develop algorithms for the automated segmentation of solar filaments. Any computational approach may be employed, including traditional image-processing techniques, machine learning methods, and state-of-the-art deep neural network architectures.

The objective of this challenge is to develop an algorithm---whether novel or existing---that generates accurate segmentation masks for individual solar filaments. For each filament, the predicted mask should capture the complete extent of the filament, including its fine-scale structures, while minimizing the inclusion of non-filament regions. The challenge emphasizes robust and precise delineation of filament morphology across a diverse set of solar observations.

# GOAL
**(Updated: Aug 7, 2026)**
  
This competition focuses on the pixel-level precision of filament segmentation masks. ~~The primary evaluation criterion is the Dice score, which quantifies the degree of overlap between the predicted and ground-truth segmentation masks (see [torchmetrics.segmentation.DiceScore](https://lightning.ai/docs/torchmetrics/stable/segmentation/dice.html))~~ **The primary evaluation criteria is the Panoptic Quality metric ([10.1109/CVPR.2019.00963](https://ieeexplore.ieee.org/abstract/document/8953237)).** In addition to segmentation accuracy, the evaluation framework incorporates penalties related to fragmentation and over-merging (e.g., one-to-many and many-to-one correspondences between predicted and ground-truth filament structures), as well as the end-to-end computational efficiency of the proposed method.
## Page: Final Submission
**(Added: Aug 7, 2026)**  

When ready, prepare the following items and submit them through [this google form](https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform?usp=dialog). The items are:
 - **A technical report**: A 4-page technical report is required. Please use [this Overleaf template](https://www.overleaf.com/read/vjztpvrnbrdh#43fbbe) for your report. More details about the content of this report are provided in the template. The final report should be submitted as a single pdf file.
 - **Code Repository**: A publicly accessible link to a Git-based repository such as GitHub, GitLab, or Bitbucket, is required. The organizing committee should access your source code without requiring additional access permissions or approval. Within this repository, a file named `requirements.txt` must include all utilized packages and their corresponding versions. Also, a jupyter notebook should illustrate the entire pipeline. The committee should not require any additional file to reproduce the submitted predictions.

All participants who wish to be considered for the final evaluation must complete the Google form. Please ensure that all information provided is accurate and complete. Any errors or omissions that prevent the judging committee from accessing the required materials or contacting the participant may result in the submission being deemed ineligible for final evaluation.
## Page: Description
#### What Are Solar Filaments?

Filaments (as shown below) are dense "clouds" of solar material suspended by magnetic field lines above photospheric neutral lines. The significance of filaments for space weather research lies in the fact that they are at the core of solar eruptions, including Coronal Mass Ejections (CMEs), solar flares, and Solar Energetic Particle (SEP) storms. An Earth-directed CME can cause enormous damage to the electric power grid, disrupt GPS systems, create radiation hazards for passengers and crew on polar flights, and be lethal to astronauts traveling outside the protective bubble provided by the Earth's magnetosphere. This is why solar filaments are a critical event type in space weather research and forecasting operations.

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3070992%2F77fe20a042912564b30d941836d26f70%2Ffilament_GONG_20240103224642Ch.png?generation=1779296142924487&alt=media" alt="an example of a solar filament" style="width:60%; height:auto;"/>

#### What Is MAGFiLO?
MAGFiLO, short for *Manually Annotated GONG Filaments from H-Alpha Observations*, is the ground-truth data of filaments' segmentation mask ([10.1038/s41597-024-03876-y](https://doi.org/10.1038/s41597-024-03876-y)). It is created for training and evaluation of machine-learning algorithms, and for evaluation of traditional computer-vision algorithms.

The illustration below shows an H-Alpha observation of the Sun, in which solar filaments have been identified by expert human annotators. Although the dataset includes additional annotations, such as bounding boxes, filament spines, and class labels, this competition focuses exclusively on the segmentation masks of solar filaments.

MAGFiLO is the testbed for all participants' models.

See the Data tab for more details.

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3070992%2F7b19d336400715aa6b69a895f6cee0ed%2Fmagfilo_segmentation_examples.png?generation=1779302394935245&alt=media" alt="an example of a MAGFiLO instances" style="width:100%; height:auto;">

#### What Are the Main Challenges?
Although recent advances in object segmentation have led to remarkable performance across many domains, solar filament segmentation remains a challenging task for several reasons:  
- **Fine-scale structures.** Accurately capturing fine filament structures, such as barbs, remains difficult. Barbs are thin, thread-like features that extend from the main body of a filament along a characteristic orientation. Their orientation relative to the filament spine carries important information about the filament's underlying magnetic field configuration.  
- **Background noise and image quality.** Distinguishing filament material (dark regions) from background structures and noise is nontrivial. Since the observations used in this competition originate from ground-based observatories, they are affected by various sources of noise and imaging artifacts. Accurately identifying small-scale filament structures while suppressing background noise remains a significant challenge.  
- **Structural continuity.** Existing segmentation algorithms often struggle to identify solar filaments as contiguous physical structures. Instead, they may produce fragmented segmentations or segment clusters of nearby dark regions ("islands"), resulting in incomplete or physically inconsistent representations of the filament morphology.
## Page: Evaluation
**(Updated: Aug 7, 2026)**  

Although participants may use other external data sources, please note that: (1) for inference of ML models, only the H-alpha images provided in the test directory will be used; and (2) the models may not use any other ground-truth metadata for training.

All submissions are judged based on the following rubric:

- Quantitative Comparison (70%):
  - ~~Mean Dice score (using [torchmetrics.segmentation.DiceScore](https://lightning.ai/docs/torchmetrics/stable/segmentation/dice.html))~~ **Panoptic Quality metric (see the Leaderboard Ranking section below for more details.)**
  - Distribution of Dice scores
  - Distribution of IoU scores
  - Distribution of one-to-many and many-to-one relations between the ground-truth and predicted segmentations.
- Qualitative Comparison (30%):
  - Detailed description of the entire pipeline (from preprocessing to final prediction, including the architecture of the utilized algorithm).
  - The apparent morphology of predicted segmentations on H-Alpha images.
  - The quality of code (modularity and documentation)

**Note:** The evaluation of the above criteria is contingent upon the availability of the source code. Please refer to the Open-Access Policy section below.
## Page: Self Evaluation
**(Added: Aug 7, 2026)**  

A self-evaluation notebook containing a few scores and plots is now available. Participants can use this notebook to have a better understanding of the effectiveness of their segmentation algorithms.

- Notebook: [https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
## Page: Leaderboard Ranking
**(Added: Aug 7, 2026)**  

The ranking on the leaderboard is one of the considerations in our evaluation (see above rubric.)

The submissions will be evaluated against the ground-truth data (MAGFiLO) using the **Panoptic Quality** metric which is defined as follows:

$$
\text{PQ}(Y, \hat{Y}) = \frac{\sum_{(\mathbf{y}, \hat{\mathbf{y}}) \in TP} \text{IoU}(\mathbf{y}, \hat{\mathbf{y}})}{|TP| + 0.5 |FP| + 0.5 |FN|}
$$

where
 - Y is a set of all ground-truth segments y,
 - Y-hat is a set of all predicted segments y-hat,
 - IoU computes the Intersection-over-Union as shown below,
 - TP, FP, and FN denote sets of all true-positive, false-positive, and false-negative cases, respectively, and
 - |.| is the set cardinality operation.

Intersection-over-Union is measured as follows:
$$
\text{IoU}(\mathbf{y}, \hat{\mathbf{y}}) = \text{IoU}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\cap(\mathbf{y}, \hat{\mathbf{y}})}{\cup{(\mathbf{y}, \hat{\mathbf{y}})}} 
                = \frac{\sum {(\mathbf{y} \odot \hat{\mathbf{y}}})}{\sum ({\mathbf{y} \oplus \hat{\mathbf{y}} \ominus \mathbf{y} \odot \hat{\mathbf{y}})}}
$$


For more information about Panoptic Quality metric, see the original paper ([10.1109/CVPR.2019.00963](https://ieeexplore.ieee.org/abstract/document/8953237)).

**Note:** There are further verification strategies addressing certain edge cases. Appropriate messages will be shown after a submission is uploaded, when/if such cases are identified.
## Page: Submission File
We expect the participants to upload a single CSV file for the entire test set, where each row corresponds to one predicted filament, as shown below.

```text
filament_id        | segmentation_rle   |
-------------------|--------------------|
20150125172714Mh_1 | "f8uSDds ... VQNC" |
20150125172714Mh_2 | "KHT%$HD ... 9>km" |
20150125172714Mh_3 | "YQNEgn1 ... BH6^" |
...                | ...                |
20170501024112Bh_1 | "HBy4d6D ... 97*D" | 
```

 - Column `filament_id` contains unique ids for each filament, e.g., `20150125172714Mh_2`.
 - Column `segmentation_rle` contains RLE counts each encoding the mask corresponding to one filament, e.g., ```^Vj02jo16I5O2O1`PNA]o1c0N19G1N11O3L01O4JYamT3```. Do not include quotations (`'` or `"`).

**Note:** You only need to store RLE Counts; no need to store RLE Size. The Size is fixed for all images: 2048 X 2048 pixels.

To convert your output (e.g., masks, polygons, etc.) to the expected format (RLE Counts), please use [the `pycocotools` Python package](https://pypi.org/project/pycocotools), specifically, the methods `annToMask`, `decodeMask`, and `encodeMask`. For more details, see [coco.py](https://github.com/ppwwyyxx/cocoapi/blob/master/PythonAPI/pycocotools/coco.py).


**Note:** The tail strings in the `filament_id` column (e.g., `_2`) is only to make the rows unique. For example, if your algorithm identifies 3 filaments in a given image with id `20150125172714Mh`, your should generate 3 rows with the following keys: `20150125172714Mh_1`, `20150125172714Mh_2`, and `20150125172714Mh_3`. As long as the tail strings render the rows unique , and the image id remains unchanged, your filament id is acceptable.  
**Note:** The number of ground-truth segmentations may also be different from the number of predicted ones. So, the evaluation method matches the predicted and ground-truth segmentations based on their actual overlap, not their index. 

The scoreboard will show to the contestants the mean Dice score for ~50% of images in the test set. The results on the remaining images will be only visible to the organizers of the competition. For a fair evaluation pipeline, the two scores should be fairly close.
## Page: Prizes
All submissions will be evaluated according to the criteria described above. The organizing committee will review the submitted solutions and determine scores for the top-3 teams. Awards will be granted across the first-, second-, and third-place categories. A total prize pool of up-to $3,000 will be distributed among the winning teams, proportional to their scores which is assigned based on the Evaluation rubric).

The winners of this competition will be announced in the IEEE BigData 2026 in December 14-17, 2026 at Phoenix, AZ.

The exact prize calculation method is shown below:
```Python
def compute_prizes(scores, total_prize=3000, temperature=2.0):
    scores = np.asarray(scores, dtype=float)
    weights = np.power(scores, temperature)
    weights = weights / np.sum(weights)
    prizes = total_prize * weights

    return prizes
```

Upon determining the winners and their exact prizes, AURA/NSO will contact the winners requesting bank information.

**Note:** Prizes will be split equally among all verified team members listed on the winning submission.

### Requirements:
- **Banking Info Requirement.** All prize recipients must provide valid banking and payment information to AURA and complete any required vendor onboarding process in accordance with AURA policies. Prize payments will be issued by AURA/NSO via direct bank transfer. Prize disbursement is contingent upon the recipient being able to receive payments from U.S. financial institutions. Failure to complete the required vendor setup or provide valid payment information will result in forfeiture of the prize.
- **Source Code Release Requirement.** Acceptance of a prize is contingent upon the winning teams providing the organizing committee with the complete source code, scripts, configuration files, and any other materials necessary to reproduce the submitted results. The submitted code must successfully reproduce the competition results using the instructions provided by the team.

By accepting a prize, the winning team grants the organizers a perpetual, worldwide, non-exclusive, royalty-free license to use, execute, reproduce, modify, and analyze the submitted code for research, educational, benchmarking, and archival purposes. This includes, but is not limited to, reproducing the reported results, evaluating and comparing submitted methods, and preparing scientific publications (e.g., survey or benchmark papers) describing the competition and the approaches submitted. The organizers will provide appropriate attribution to the authors when the submitted methods or code are discussed or referenced in such publications. 

- **Technical Report Requirement.** Acceptance of a prize is contingent upon the winning team submitting a concise technical report describing its solution. The report must document the complete data processing pipeline, including data preprocessing, feature engineering (if applicable), model architecture, training procedure, post-processing steps, and any other components necessary to understand and reproduce the submitted results. The report should be sufficiently detailed to enable the organizing committee to evaluate the methodology and independently reproduce the reported performance.  


**Note:** The organizing committee will not claim authorship of, or take credit for, any submitted code or methods. All intellectual contributions will remain attributed to their original authors, and appropriate credit will be given in any resulting publications or other uses of the submitted materials.  
## Page: Open-Access Policy
All participants wishing to be considered for the final evaluation must maintain a Git repository containing the source code of their solution. The repository must be made publicly accessible immediately upon the close of the competition and must remain publicly accessible at least until when the winners are announced. Please refer to the competition end date and time for the submission deadline.

The quality of the submitted code—including its modularity, reproducibility, and documentation—will be considered as part of the final evaluation and may affect a team's overall score. Please refer to the Evaluation section for additional details.

## Page: Organizers
- Azim Ahmadzadeh, Ph.D., University of Missouri–St. Louis ([www.azim-a.com/](https://www.azim-a.com/))
- Dustin J. Kempton, Ph.D., Georgia State University
- Qin Li, Ph.D., New Jersey Institute of Technology
- Alexei A. Pevtsov, Ph.D., National Solar Observatory
## Page: Sponsor
This competition is sponsored by the U.S. National Science Foundation (NSF) National Solar Observatory (NSO). The NSO is the United States' national center for ground-based solar physics and advances research, education, and public outreach in solar science. The NSO is operated by the Association of Universities for Research in Astronomy (AURA) under a cooperative agreement with the NSF Division of Astronomical Sciences.

For more information about the National Solar Observatory, please visit [www.nso.edu](https://nso.edu/).

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3070992%2F92e6fcb0cff5365b607c6b6057ee6544%2Fnso-aura_logos.png?generation=1783377127838047&alt=media)

The organization of the Filament Segmentation Challenge 2026 is partially supported by the U.S. National Science Foundation (NSF) under Grant No. 2209912 and 2433781, directorate for Computer and Information Science and Engineering (CSE), and Office of Advanced Cyberinfrastructure (OAC), and Grant No. 2511630, AST Division Of Astronomical Sciences and MPS Directorate for Mathematical and Physical Sciences.
## Page: Acknowledgment
This work utilizes GONG data obtained by the NSO Integrated Synoptic Program, managed by the National Solar Observatory, which is operated by the Association of Universities for Research in Astronomy (AURA), Inc. under a cooperative agreement with the National Science Foundation and with contribution from the National Oceanic and Atmospheric Administration. The GONG network of instruments is hosted by the Big Bear Solar Observatory, High Altitude Observatory, Learmonth Solar Observatory, Udaipur Solar Observatory, Instituto de Astrofísica de Canarias, and Cerro Tololo Interamerican Observatory.
