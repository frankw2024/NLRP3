# Flowcharts for CD28 Project Python Files

This document contains flowcharts for all Python scripts in the CD28 project, providing a visual representation of their execution flow and logic.

---

## 1. get_trainingSet.py

**Purpose**: Fetch all NLRP3 bioactivities from ChEMBL and write a CSV file.

### Main Flowchart

```
                    ┌─────┐
                    │Start│
                    └──┬──┘
                       │
                       ▼
        ┌───────────────────────────────┐
        │Search ChEMBL for NLRP3        │
        │target IDs                     │
        └───────────────┬───────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │ Targets Found?  │
              └─────┬───────┬───┘
                    │       │
              ┌─────┘       └─────┐
              │ No                │ Yes
              ▼                   ▼
    ┌─────────────────┐   ┌──────────────┐
    │Print Error &    │   │Rank Targets  │
    │Exit             │   │by Organism   │
    └────────┬────────┘   │& Type        │
             │            └──────┬───────┘
             │                   │
             │                   ▼
             │          ┌──────────────┐
             │          │Select Best   │
             │          │Target        │
             │          └──────┬───────┘
             │                 │
             │                 ▼
             │      ┌──────────────────────┐
             │      │Fetch All Activities  │
             │      │via Pagination        │
             │      └──────┬───────────────┘
             │             │
             │             ▼
             │     ┌──────────────┐
             │     │Normalize     │
             │     │Activity Rows │
             │     └──────┬───────┘
             │            │
             │            ▼
             │    ┌──────────────┐
             │    │Deduplicate   │
             │    │by Key Fields │
             │    └──────┬───────┘
             │           │
             │           ▼
             │   ┌──────────────┐
             │   │Extract Unique│
             │   │Molecule IDs  │
             │   └──────┬───────┘
             │          │
             │          ▼
             │  ┌──────────────────────┐
             │  │Fetch Canonical       │
             │  │SMILES for Molecules  │
             │  └──────┬───────────────┘
             │         │
             │         ▼
             │ ┌──────────────┐
             │ │Attach SMILES │
             │ │to Activity   │
             │ │Rows          │
             │ └──────┬───────┘
             │        │
             │        ▼
             │┌──────────────┐
             ││Write to CSV  │
             ││nlrp3_chembl_ │
             ││activities.csv│
             │└──────┬───────┘
             │       │
             └───────┴───┐
                         │
                         ▼
                    ┌────┘
                    │End
                    └────
```

### Function: find_nlrp3_target_ids()

```
    ┌─────────────────────────┐
    │Start: find_nlrp3_target │
    │_ids()                   │
    └────────────┬────────────┘
                 │
                 ▼
        ┌────────────────┐
        │Search by Target│
        │Name            │
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │Search by Gene  │
        │Symbol          │
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │Remove Duplicate│
        │Target IDs      │
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │Return List of  │
        │Target IDs      │
        └────────┬───────┘
                 │
                 ▼
            ┌────┘
            │End
            └────
```

### Function: fetch_all_activities()

```
    ┌─────────────────────────┐
    │Start: fetch_all_        │
    │activities()             │
    └────────────┬────────────┘
                 │
                 ▼
    ┌────────────────────────┐
    │Initialize: offset=0,   │
    │activities=[]           │
    └────────────┬───────────┘
                 │
                 ▼
            ┌─────────┐
            │More     │◄────────────────────┐
            │Pages?   │                     │
            └───┬───┬─┘                     │
                │   │                       │
          ┌─────┘   └─────┐                │
          │ Yes           │ No              │
          ▼               ▼                 │
    ┌─────────────┐ ┌──────────────┐       │
    │GET /activity│ │Return        │       │
    │endpoint with│ │Activities    │       │
    │pagination   │ │List          │       │
    └──────┬──────┘ └──────┬───────┘       │
           │               │               │
           ▼               │               │
    ┌──────────────┐       │               │
    │Status 200?   │       │               │
    └───┬──────┬───┘       │               │
        │      │           │               │
    ┌───┘      └───┐       │               │
    │ No           │ Yes   │               │
    ▼              ▼       │               │
┌─────────┐  ┌──────────┐ │               │
│Retries  │  │Extract   │ │               │
│< 5?     │  │Activities│ │               │
└───┬───┬─┘  │from      │ │               │
    │   │    │Response  │ │               │
┌───┘   └──┐ └────┬─────┘ │               │
│Yes       │No    │       │               │
▼          ▼      │       │               │
┌──────┐ ┌─────┐ │       │               │
│Sleep │ │Raise│ │       │               │
│&     │ │Runt │ │       │               │
│Retry │ │imeEr│ │       │               │
└───┬──┘ └──┬──┘ │       │               │
    │       │    │       │               │
    └───────┘    └───────┘               │
                       │                 │
                       ▼                 │
                  ┌──────────┐           │
                  │Append to │           │
                  │Activities│           │
                  │List      │           │
                  └────┬─────┘           │
                       │                 │
                       ▼                 │
                  ┌──────────┐           │
                  │Update    │           │
                  │Offset    │           │
                  └────┬─────┘           │
                       │                 │
                       ▼                 │
                  ┌──────────┐           │
                  │Sleep     │           │
                  │RATE_SLEEP│           │
                  └────┬─────┘           │
                       │                 │
                       ▼                 │
                  ┌──────────┐           │
                  │Offset >= │           │
                  │Total?    │           │
                  └────┬─────┘           │
                       │                 │
                       └─────────────────┘
```

### Function: fetch_smiles_for_molecules()

```
    ┌─────────────────────────┐
    │Start: fetch_smiles_for  │
    │_molecules()             │
    └────────────┬────────────┘
                 │
                 ▼
    ┌────────────────────────┐
    │Initialize id_to_smiles │
    │dict                    │
    └────────────┬───────────┘
                 │
                 ▼
            ┌─────────┐
            │For Each │◄──────────┐
            │Molecule │           │
            │ID       │           │
            └───┬───┬─┘           │
                │   │             │
          ┌─────┘   └─────┐       │
          │ Next          │ Done  │
          ▼               ▼       │
    ┌─────────────┐ ┌──────────┐ │
    │GET /molecule│ │Return    │ │
    │endpoint     │ │id_to_    │ │
    └──────┬──────┘ │smiles    │ │
           │        └────┬─────┘ │
           ▼             │       │
    ┌──────────────┐     │       │
    │Extract       │     │       │
    │Canonical     │     │       │
    │SMILES        │     │       │
    └──────┬───────┘     │       │
           │             │       │
           ▼             │       │
    ┌──────────────┐     │       │
    │Store in      │     │       │
    │Dictionary    │     │       │
    └──────┬───────┘     │       │
           │             │       │
           ▼             │       │
    ┌──────────────┐     │       │
    │Sleep         │     │       │
    │RATE_SLEEP    │     │       │
    └──────┬───────┘     │       │
           │             │       │
           └─────────────┘       │
                                 │
                                 ▼
                            ┌────┘
                            │End
                            └────
```

---

## 2. HTS.py

**Purpose**: Train an ensemble model for high-throughput screening using molecular features and ChemBERTa embeddings.

### Main Flowchart

```
                    ┌─────┐
                    │Start│
                    └──┬──┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │Load library.csv &            │
        │positives.csv                 │
        └──────────────┬───────────────┘
                       │
                       ▼
            ┌──────────────────┐
            │Create Binary     │
            │Labels            │
            └──────────┬───────┘
                       │
                       ▼
            ┌──────────────────┐
            │Check Class       │
            │Balance?          │
            └────┬─────────┬───┘
                 │         │
          ┌──────┘         └──────┐
          │ Imbalanced            │ OK
          ▼                       ▼
    ┌──────────┐         ┌───────────────┐
    │Print     │         │Generate RDKit │
    │Warning   │         │Features       │
    └────┬─────┘         └───────┬───────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │Initialize Stratified   │
        │K-Fold CV               │
        └────────────┬───────────┘
                     │
                     ▼
            ┌───────────────┐
            │For Each Feature│◄───────────────┐
            │Method          │                │
            └────┬───────┬───┘                │
                 │       │                    │
           ┌─────┘       └─────┐             │
           │ Next Method       │ All Methods │
           ▼                   ▼             │
    ┌──────────────┐   ┌──────────────┐     │
    │For Each Fold │◄──│Build Stacked │     │
    │              │───│Ensemble      │     │
    └────┬──────┬──┘   └──────┬───────┘     │
         │      │              │             │
    ┌────┘      └─────┐        │             │
    │ Next Fold       │        │             │
    │ All Folds       │        │             │
    ▼                 ▼        │             │
┌──────────┐    ┌──────────┐  │             │
│Split     │    │Apply     │  │             │
│Train/Val │    │Feature   │  │             │
└────┬─────┘    │Selection │  │             │
     │          └────┬─────┘  │             │
     │               │        │             │
     ▼               ▼        │             │
┌──────────┐    ┌──────────┐  │             │
│Create    │    │Initialize│  │             │
│PyTorch   │    │Model     │  │             │
│Datasets  │    └────┬─────┘  │             │
└────┬─────┘         │        │             │
     │               │        │             │
     ▼               ▼        │             │
┌──────────┐    ┌──────────┐  │             │
│Train     │    │Validate  │  │             │
│Model with│    │Model     │  │             │
│Early     │    └────┬─────┘  │             │
│Stopping  │         │        │             │
└────┬─────┘         │        │             │
     │               │        │             │
     ▼               ▼        │             │
┌──────────┐    ┌──────────┐  │             │
│Save Model│    │          │  │             │
│State     │────┘          │  │             │
└──────────┘               │  │             │
                           │  │             │
                           └──┘             │
                                            │
                                            ▼
                                    ┌──────────────┐
                                    │Evaluate Final│
                                    │Performance   │
                                    └──────┬───────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │Save Model &  │
                                    │Predictions   │
                                    └──────┬───────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │Generate      │
                                    │Performance   │
                                    │Plots         │
                                    └──────┬───────┘
                                           │
                                           ▼
                                        ┌──┘
                                        │End
                                        └──
```

### Feature Generation Flow

```
    ┌──────────────────────┐
    │Start: Generate       │
    │Features              │
    └──────────┬───────────┘
               │
               ▼
        ┌──────────┐
        │For Each  │◄─────────────────────┐
        │SMILES    │                      │
        └───┬────┬─┘                      │
            │    │                        │
      ┌─────┘    └─────┐                  │
      │ Next           │ Done             │
      ▼                ▼                  │
┌──────────────┐ ┌──────────────┐        │
│Parse SMILES  │ │Return        │        │
│to Molecule   │ │Features      │        │
└──────┬───────┘ │Array         │        │
       │         └──────┬───────┘        │
       ▼                │                │
  ┌─────────┐           │                │
  │Valid    │           │                │
  │Molecule?│           │                │
  └───┬───┬─┘           │                │
      │   │             │                │
  ┌───┘   └───┐         │                │
  │ No        │ Yes     │                │
  ▼           ▼         │                │
┌──────┐ ┌──────────┐  │                │
│Return│ │Generate  │  │                │
│Zero  │ │Morgan    │  │                │
│Feat  │ │Fingerprin│  │                │
│ures  │ │t         │  │                │
└───┬──┘ └────┬─────┘  │                │
    │         │        │                │
    │         ▼        │                │
    │  ┌──────────────┐│                │
    │  │Generate MACCS││                │
    │  │Keys          ││                │
    │  └──────┬───────┘│                │
    │         │        │                │
    │         ▼        │                │
    │  ┌──────────────┐│                │
    │  │Generate      ││                │
    │  │Physicochemical│                │
    │  │Descriptors   ││                │
    │  └──────┬───────┘│                │
    │         │        │                │
    │         ▼        │                │
    │  ┌──────────────┐│                │
    │  │Concatenate   ││                │
    │  │All Features  ││                │
    │  └──────┬───────┘│                │
    │         │        │                │
    │         ▼        │                │
    │  ┌──────────────┐│                │
    │  │NaN or Inf?   ││                │
    │  └───┬───────┬──┘│                │
    │      │       │   │                │
    │  ┌───┘       └───┐                │
    │  │ Yes           │ No             │
    │  ▼               ▼                │
    │┌──────┐     ┌──────────┐         │
    ││Replace│     │Append to │         │
    ││with   │     │Features  │         │
    ││Zeros  │     │Array     │         │
    │└───┬──┘     └────┬─────┘         │
    │    │             │               │
    │    └──────┬──────┘               │
    │           │                      │
    │           └──────────────────────┘
    │
    ▼
┌────┘
│End
└────
```

### Training Loop Flow

```
    ┌──────────────────────┐
    │Start: Training Loop  │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────────┐
    │Initialize: best_auc=0,   │
    │no_improve=0              │
    └──────────┬───────────────┘
               │
               ▼
        ┌──────────┐
        │For Each  │◄────────────────────────────┐
        │Epoch     │                             │
        └───┬────┬─┘                             │
            │    │                               │
      ┌─────┘    └─────┐                         │
      │ Next           │ Done                    │
      ▼                ▼                         │
┌──────────────┐ ┌──────────────┐               │
│Set Model to  │ │Return Best   │               │
│Train Mode    │ │Model         │               │
└──────┬───────┘ └──────┬───────┘               │
       │                │                       │
       ▼                │                       │
  ┌──────────┐          │                       │
  │For Each  │◄─────────┐                       │
  │Batch     │          │                       │
  └───┬────┬─┘          │                       │
      │    │            │                       │
  ┌───┘    └─────┐      │                       │
  │ Next         │ Done │                       │
  ▼              ▼      │                       │
┌──────────┐ ┌──────────┐                      │
│Forward   │ │Set Model │                      │
│Pass      │ │to Eval   │                      │
└────┬─────┘ │Mode      │                      │
     │       └────┬─────┘                      │
     ▼            │                            │
┌──────────────┐  │                            │
│NaN           │  │                            │
│Predictions?  │  │                            │
└───┬───────┬──┘  │                            │
    │       │     │                            │
┌───┘       └───┐ │                            │
│ Yes           │ │                            │
│ No            │ │                            │
▼               ▼ │                            │
┌──────────┐ ┌──────────┐                     │
│Skip Batch│ │Calculate │                     │
└────┬─────┘ │Loss      │                     │
     │       └────┬─────┘                     │
     │            │                           │
     │            ▼                           │
     │      ┌──────────────┐                 │
     │      │Backward Pass │                 │
     │      └──────┬───────┘                 │
     │             │                         │
     │             ▼                         │
     │      ┌──────────────┐                 │
     │      │Clip Gradients│                 │
     │      └──────┬───────┘                 │
     │             │                         │
     │             ▼                         │
     │      ┌──────────────┐                 │
     │      │Update Weights│                 │
     │      └──────┬───────┘                 │
     │             │                         │
     │             └─────────────────────────┘
     │                                         │
     ▼                                         │
  ┌──────────┐                                │
  │For Each  │◄───────────────────────────────┐
  │Val Batch │                                │
  └───┬────┬─┘                                │
      │    │                                  │
  ┌───┘    └─────┐                            │
  │ Next         │ Done                       │
  ▼              ▼                            │
┌──────────┐ ┌──────────────┐                │
│Forward   │ │Calculate AUC │                │
│Pass      │ │& AP          │                │
└────┬─────┘ └──────┬───────┘                │
     │              │                        │
     ▼              │                        │
┌──────────────┐    │                        │
│NaN           │    │                        │
│Predictions?  │    │                        │
└───┬───────┬──┘    │                        │
    │       │       │                        │
┌───┘       └───┐   │                        │
│ Yes           │   │                        │
│ No            │   │                        │
▼               ▼   │                        │
┌──────────┐ ┌──────────┐                   │
│Skip Batch│ │Collect   │                   │
└────┬─────┘ │Predictions│                  │
     │       └────┬─────┘                   │
     │            │                         │
     │            └─────────────────────────┘
     │                                        │
     │                                        │
     ▼                                        │
┌──────────────┐                             │
│Update        │                             │
│Training      │                             │
│History       │                             │
└──────┬───────┘                             │
       │                                     │
       ▼                                     │
  ┌──────────────┐                           │
  │AUC >         │                           │
  │best_auc?     │                           │
  └───┬───────┬──┘                           │
      │       │                             │
  ┌───┘       └───┐                         │
  │ Yes           │ No                      │
  ▼               ▼                         │
┌──────────┐ ┌──────────┐                  │
│Save Best │ │Increment │                  │
│Model     │ │no_improve│                  │
│State     │ └────┬─────┘                  │
└────┬─────┘      │                        │
     │            │                        │
     ▼            ▼                        │
┌──────────┐ ┌──────────────┐             │
│Reset     │ │no_improve >= │             │
│no_improve│ │patience?     │             │
│=0        │ └───┬───────┬──┘             │
└────┬─────┘     │       │                │
     │       ┌───┘       └───┐            │
     │       │ Yes           │ No         │
     │       ▼               │            │
     │  ┌──────────┐         │            │
     │  │Break     │         │            │
     │  │Early     │         │            │
     │  │Stopping  │         │            │
     │  └────┬─────┘         │            │
     │       │               │            │
     │       └───────────────┘            │
     │                                    │
     └────────────────────────────────────┘
```

### Ensemble Building Flow

```
    ┌──────────────────────┐
    │Start: Build Ensemble │
    └──────────┬───────────┘
               │
               ▼
    ┌────────────────────────┐
    │Initialize Prediction   │
    │Arrays                  │
    └──────────┬─────────────┘
               │
               ▼
        ┌──────────────┐
        │For Each      │◄───────────────────────┐
        │Feature Method│                        │
        └───┬───────┬──┘                        │
            │       │                          │
      ┌─────┘       └─────┐                    │
      │ Next Method       │ All Methods        │
      ▼                   ▼                    │
  ┌──────────┐     ┌──────────────┐           │
  │For Each  │◄────│Final Ensemble│           │
  │Model     │─────│Average       │           │
  └───┬────┬─┘     └──────┬───────┘           │
      │    │              │                   │
  ┌───┘    └─────┐        │                   │
  │ Next         │ Done   │                   │
  ▼              ▼        │                   │
┌──────────┐ ┌──────────┐│                   │
│For Each  │◄│Average   ││                   │
│Sample    │─│Predictions│                   │
└───┬────┬─┘ └────┬─────┘│                   │
    │    │        │      │                   │
┌───┘    └─────┐  │      │                   │
│ Next         │  │      │                   │
│ Done         │  │      │                   │
▼              ▼  │      │                   │
┌──────────┐ ┌──────────┐│                   │
│Apply     │ │Load Model││                   │
│Feature   │ │State     ││                   │
│Selection │ └────┬─────┘│                   │
└────┬─────┘      │      │                   │
     │            │      │                   │
     ▼            ▼      │                   │
┌──────────┐ ┌──────────┐│                   │
│Generate  │ │NaN or Inf││                   │
│Prediction│ │?         ││                   │
└────┬─────┘ └───┬────┬─┘│                   │
     │           │    │  │                   │
     │       ┌───┘    └───┐                 │
     │       │ Yes         │ No              │
     │       ▼             ▼                 │
     │  ┌──────────┐ ┌──────────┐           │
     │  │Skip      │ │Accumulate│           │
     │  │Prediction│ │Prediction│           │
     │  └────┬─────┘ └────┬─────┘           │
     │       │            │                 │
     │       └──────┬─────┘                 │
     │              │                       │
     │              └───────────────────────┘
     │
     ▼
┌──────────────┐
│Clip to [0,1] │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Save Ensemble │
│Model         │
└──────┬───────┘
       │
       ▼
    ┌──┘
    │End
    └──
```

---

## 3. HTSOracle.py

**Purpose**: Streamlit web application for molecular activity prediction using the trained ensemble model.

### Main Application Flow

```
    ┌──────────────────────┐
    │Start: Streamlit App  │
    └──────────┬───────────┘
               │
               ▼
    ┌────────────────────────┐
    │Configure Page Settings │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │Create Sidebar with     │
    │Options                 │
    └──────────┬─────────────┘
               │
               ▼
        ┌──────────┐
        │File      │◄──────────────┐
        │Uploaded? │               │
        └───┬────┬─┘               │
            │    │                 │
      ┌─────┘    └─────┐           │
      │ No             │ Yes       │
      ▼                ▼           │
┌──────────┐     ┌──────────────┐ │
│Wait for  │     │Model File    │ │
│Upload    │     │Exists?       │ │
└────┬─────┘     └───┬───────┬──┘ │
     │               │       │    │
     └───────────────┘   ┌───┘    └───┐
                         │ No         │ Yes
                         ▼            ▼
                    ┌──────────┐ ┌──────────┐
                    │Show      │ │Load      │
                    │Warning   │ │Ensemble  │
                    └────┬─────┘ │Model     │
                         │       └────┬─────┘
                         │            │
                         └──────┬─────┘
                                │
                                ▼
                        ┌──────────────┐
                        │Read Uploaded │
                        │CSV           │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │Find SMILES   │
                        │Column        │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │Extract SMILES│
                        │List          │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │Validate      │
                        │SMILES Strings│
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │Generate RDKit│
                        │Features      │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │Prediction    │
                        │Mode?         │
                        └───┬───────┬──┘
                            │       │
                        ┌───┘       └───┐
                        │ Simple        │ Ensemble
                        ▼               ▼
                ┌──────────────┐ ┌──────────────┐
                │Simple Drug-  │ │Enhanced      │
                │Likeness      │ │Ensemble      │
                │Predictions   │ │Predictions   │
                └──────┬───────┘ └──────┬───────┘
                       │                │
                       └────────┬───────┘
                                │
                                ▼
                        ┌──────────────┐
                        │Create Results│
                        │DataFrame     │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │Display       │
                        │Summary       │
                        │Statistics    │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │Display       │
                        │Distribution  │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │Show          │
                        │Properties?   │
                        └───┬───────┬──┘
                            │       │
                        ┌───┘       └───┐
                        │ Yes           │ No
                        ▼               ▼
                ┌──────────────┐ ┌──────────────┐
                │Analyze       │ │Show Method   │
                │Molecular     │ │Comparison?   │
                │Properties    │ └───┬───────┬──┘
                └──────┬───────┘     │       │
                       │         ┌───┘       └───┐
                       │         │ Yes           │ No
                       │         ▼               ▼
                       │  ┌──────────────┐ ┌──────────────┐
                       │  │Compare       │ │Display       │
                       │  │Feature       │ │Results Table │
                       │  │Methods       │ └──────┬───────┘
                       │  └──────┬───────┘        │
                       │         │                │
                       └─────────┴────────────────┘
                                 │
                                 ▼
                        ┌──────────────┐
                        │Provide       │
                        │Download      │
                        │Button        │
                        └──────┬───────┘
                               │
                               ▼
                            ┌──┘
                            │End
                            └──
```

### Prediction Generation Flow

```
    ┌──────────────────────┐
    │Start: Generate       │
    │Predictions           │
    └──────────┬───────────┘
               │
               ▼
        ┌──────────┐
        │Prediction│
        │Mode?     │
        └───┬────┬─┘
            │    │
      ┌─────┘    └─────┐
      │ Simple         │ Ensemble
      ▼                ▼
┌──────────────┐ ┌──────────────┐
│Use Drug-     │ │Model Loaded? │
│Likeness      │ └───┬───────┬──┘
│Based         │     │       │
│Predictions   │ ┌───┘       └───┐
└──────┬───────┘ │ No            │ Yes
       │         ▼               ▼
       │    ┌──────────┐  ┌──────────────┐
       │    │Use       │  │For Each      │◄──────────┐
       │    │Fallback  │  │Feature Method│           │
       │    │Method    │  └───┬───────┬──┘           │
       │    └────┬─────┘      │       │             │
       │         │        ┌───┘       └───┐         │
       │         │        │ Next Method   │ All     │
       │         │        ▼               ▼ Methods │
       │         │    ┌──────────┐  ┌──────────────┐│
       │         │    │For Each  │◄─│Final Ensemble││
       │         │    │Model     │──│Average       ││
       │         │    └───┬────┬─┘  └──────┬───────┘│
       │         │        │    │           │        │
       │         │    ┌───┘    └─────┐     │        │
       │         │    │ Next         │ Done│        │
       │         │    ▼              ▼     │        │
       │         │┌──────────┐ ┌──────────┐│        │
       │         ││For Each  │◄│Average   ││        │
       │         ││Sample    │─│Method    ││        │
       │         │└───┬────┬─┘ │Predictions│        │
       │         │    │    │   └────┬─────┘│        │
       │         │┌───┘    └─────┐  │      │        │
       │         ││ Next         │  │      │        │
       │         ││ Done         │  │      │        │
       │         │▼              ▼  │      │        │
       │         │┌──────────┐ ┌──────────┐│        │
       │         ││Apply     │ │Load Model││        │
       │         ││Feature   │ │State     ││        │
       │         ││Selection │ └────┬─────┘│        │
       │         │└────┬─────┘      │      │        │
       │         │     │            │      │        │
       │         │     ▼            ▼      │        │
       │         │┌──────────┐ ┌──────────┐│        │
       │         ││Generate  │ │NaN or Inf││        │
       │         ││Method-   │ │?         ││        │
       │         ││Specific  │ └───┬────┬─┘│        │
       │         ││Prediction│     │    │  │        │
       │         │└────┬─────┘ ┌───┘    └───┐      │
       │         │     │       │ Yes         │ No   │
       │         │     │       ▼             ▼      │
       │         │     │  ┌──────────┐ ┌──────────┐│
       │         │     │  │Skip      │ │Accumulate││
       │         │     │  └────┬─────┘ └────┬─────┘│
       │         │     │       │            │      │
       │         │     │       └──────┬─────┘      │
       │         │     │              │            │
       │         │     │              └────────────┘
       │         │     │
       │         │     │
       │         └─────┘
       │
       ▼
┌──────────────┐
│Clip to [0,1] │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Return        │
│Predictions   │
└──────┬───────┘
       │
       ▼
    ┌──┘
    │End
    └──
```

### Feature Generation Flow (HTSOracle)

```
    ┌──────────────────────┐
    │Start: Generate       │
    │Features              │
    └──────────┬───────────┘
               │
               ▼
    ┌────────────────────────┐
    │Initialize Features List│
    └──────────┬─────────────┘
               │
               ▼
        ┌──────────┐
        │For Each  │◄─────────────────────┐
        │SMILES    │                      │
        └───┬────┬─┘                      │
            │    │                        │
      ┌─────┘    └─────┐                  │
      │ Next           │ Done             │
      ▼                ▼                  │
┌──────────────┐ ┌──────────────┐        │
│Update        │ │Convert to    │        │
│Progress?     │ │NumPy Array   │        │
└───┬───────┬──┘ └──────┬───────┘        │
    │       │           │                │
┌───┘       └───┐       │                │
│ Yes           │ No    │                │
▼               ▼       │                │
┌──────────┐ ┌──────────┐│                │
│Show      │ │Parse     ││                │
│Progress  │ │SMILES    ││                │
│Message   │ └────┬─────┘│                │
└────┬─────┘      │      │                │
     │            │      │                │
     └────────────┘      │                │
                         ▼                │
                    ┌─────────┐           │
                    │Valid?   │           │
                    └───┬───┬─┘           │
                        │   │             │
                    ┌───┘   └───┐         │
                    │ No        │ Yes     │
                    ▼           ▼         │
                ┌──────┐ ┌──────────────┐│
                │Return│ │Generate      ││
                │Zero  │ │Morgan FP     ││
                │Feat  │ └──────┬───────┘│
                │ures  │        │        │
                └───┬──┘        │        │
                    │           │        │
                    │           ▼        │
                    │    ┌──────────────┐│
                    │    │Generate MACCS││
                    │    │Keys          ││
                    │    └──────┬───────┘│
                    │           │        │
                    │           ▼        │
                    │    ┌──────────────┐│
                    │    │Generate      ││
                    │    │Physicochemical││
                    │    │Descriptors   ││
                    │    └──────┬───────┘│
                    │           │        │
                    │           ▼        │
                    │    ┌──────────────┐│
                    │    │Concatenate   ││
                    │    │Features      ││
                    │    └──────┬───────┘│
                    │           │        │
                    │           ▼        │
                    │    ┌──────────────┐│
                    │    │NaN or Inf?   ││
                    │    └───┬───────┬──┘│
                    │        │       │   │
                    │    ┌───┘       └───┐
                    │    │ Yes           │ No
                    │    ▼               ▼
                    │┌──────┐     ┌──────────┐
                    ││Replace│     │Append    │
                    ││with   │     │Features  │
                    ││Zeros  │     └────┬─────┘
                    │└───┬──┘           │
                    │    │              │
                    │    └──────┬───────┘
                    │           │
                    │           └──────────────────┘
                    │
                    ▼
            ┌──────────────┐
            │Final NaN/Inf │
            │Check?        │
            └───┬───────┬──┘
                │       │
            ┌───┘       └───┐
            │ Yes           │ No
            ▼               ▼
    ┌──────────────┐ ┌──────────────┐
    │Replace with  │ │Return        │
    │Zeros         │ │Features      │
    └──────┬───────┘ └──────┬───────┘
           │                │
           └────────┬───────┘
                    │
                    ▼
                 ┌──┘
                 │End
                 └──
```

### Visualization Flow

```
    ┌──────────────────────┐
    │Start: Create         │
    │Visualizations        │
    └──────────┬───────────┘
               │
               ▼
        ┌──────────┐
        │Interactive│
        │Mode?     │
        └───┬────┬─┘
            │    │
      ┌─────┘    └─────┐
      │ Yes            │ No
      ▼                ▼
┌──────────────┐ ┌──────────────┐
│Create Plotly │ │Create        │
│Dashboard     │ │Matplotlib    │
└──────┬───────┘ │Plots         │
       │         └──────┬───────┘
       │                │
       ▼                ▼
┌──────────────┐ ┌──────────────┐
│Prediction    │ │Molecular     │
│Score         │ │Weight        │
│Distribution  │ │Histogram     │
└──────┬───────┘ └──────┬───────┘
       │                │
       ▼                ▼
┌──────────────┐ ┌──────────────┐
│Top 20        │ │LogP          │
│Compounds     │ │Histogram     │
└──────┬───────┘ └──────┬───────┘
       │                │
       ▼                ▼
┌──────────────┐ ┌──────────────┐
│MW vs LogP    │ │QED           │
│Scatter       │ │Histogram     │
└──────┬───────┘ └──────┬───────┘
       │                │
       ▼                ▼
┌──────────────┐ ┌──────────────┐
│QED vs Score  │ │TPSA          │
│              │ │Histogram     │
└──────┬───────┘ └──────┬───────┘
       │                │
       ▼                ▼
┌──────────────┐ ┌──────────────┐
│Display Plotly│ │Display       │
│Chart         │ │Matplotlib    │
└──────┬───────┘ │Figure        │
       │         └──────┬───────┘
       │                │
       └────────┬───────┘
                │
                ▼
             ┌──┘
             │End
             └──
```

---

## Key Components Summary

### Data Flow Overview

```
get_trainingSet.py      CSV Files              HTS.py            enhanced_ensemble_model.pkl
(Fetch ChEMBL Data)  ──► (nlrp3_chembl_      ──► (Train Models) ──► (Trained Model)
                        activities.csv)

                          │
                          │
                          ▼
                  HTSOracle.py              Predictions & Visualizations
                  (Web Application)    ──► (Output)
```

### Error Handling Strategy

All three scripts implement comprehensive error handling:

1. **get_trainingSet.py**: Retry logic for API calls, graceful handling of missing data
2. **HTS.py**: Extensive try-except blocks, NaN/Inf checks, fallback values
3. **HTSOracle.py**: User-friendly error messages, fallback prediction methods, debug mode

### Feature Engineering Pipeline

```
                SMILES String
                      │
                      ▼
            RDKit Molecule Parsing
                      │
         ┌────────────┼────────────┐
         │            │            │
         ▼            ▼            ▼
    Morgan FP    MACCS Keys   PhysChem
   (2048 bits)  (167 bits)  (15 features)
         │            │            │
         └────────────┼────────────┘
                      │
                      ▼
             Combine Features
             (2230 total)
                      │
                      ▼
              Feature Selection
                      │
         ┌────────────┼────────────┐
         │            │            │
         ▼            ▼            ▼
      LASSO        PCA        Mutual Info
    Selection  Dimensionality  Selection
               Reduction
         │            │            │
         └────────────┼────────────┘
                      │
                      ▼
                Final Features
```

---

## 4. HTS_3D.py

**Purpose**: Train a 3D-aware cross-attention model for high-throughput screening that combines ChemBERTa embeddings, RDKit 3D conformers, and protein pocket representations.

### Main Flowchart

```
                    ┌─────┐
                    │Start│
                    └──┬──┘
                       │
                       ▼
            ┌──────────────────┐
            │CSV File Exists?  │
            └───┬───────────┬──┘
                │           │
          ┌─────┘           └─────┐
          │ No                    │ Yes
          ▼                       ▼
    ┌──────────────┐     ┌──────────────┐
    │Print Error & │     │Load Activity │
    │Exit          │     │Table from CSV│
    └──────┬───────┘     └──────┬───────┘
           │                    │
           │                    ▼
           │            ┌──────────────┐
           │            │Normalize     │
           │            │Units to nM   │
           │            └──────┬───────┘
           │                   │
           │                   ▼
           │           ┌──────────────┐
           │           │Create Binary │
           │           │Labels:       │
           │           │<= 1000 nM    │
           │           └──────┬───────┘
           │                  │
           │                  ▼
           │          ┌──────────────┐
           │          │Deduplicate by│
           │          │Molecule ID   │
           │          └──────┬───────┘
           │                 │
           │                 ▼
           │         ┌──────────────┐
           │         │Tokenize      │
           │         │SMILES with   │
           │         │ChemBERTa     │
           │         └──────┬───────┘
           │                │
           │                ▼
           │        ┌──────────────┐
           │        │Generate RDKit│
           │        │2D Features   │
           │        └──────┬───────┘
           │               │
           │               ▼
           │      ┌──────────────┐
           │      │GPU-Accelerated│
           │      │Feature       │
           │      │Scaling       │
           │      └──────┬───────┘
           │             │
           │             ▼
           │     ┌──────────────┐
           │     │Generate 3D   │
           │     │Conformer     │
           │     │Features      │
           │     └──────┬───────┘
           │            │
           │            ▼
           │    ┌──────────────┐
           │    │Create        │
           │    │HTS3DDataset  │
           │    └──────┬───────┘
           │           │
           │           ▼
           │   ┌──────────────┐
           │   │Initialize    │
           │   │Stratified    │
           │   │K-Fold CV:    │
           │   │3 folds       │
           │   └──────┬───────┘
           │          │
           │          ▼
           │     ┌──────────┐
           │     │For Each  │◄────────────────────┐
           │     │Fold      │                     │
           │     └───┬────┬─┘                     │
           │         │    │                       │
           │     ┌───┘    └─────┐                │
           │     │ Next Fold    │ All Folds      │
           │     ▼              ▼                │
           │┌──────────┐  ┌──────────────┐      │
           ││Split     │  │Save Best     │      │
           ││Train/Val │  │Model to .pt  │      │
           ││Indices   │  └──────┬───────┘      │
           │└────┬─────┘         │              │
           │     │               │              │
           │     ▼               │              │
           │┌──────────┐         │              │
           ││Create    │         │              │
           ││DataLoaders│        │              │
           │└────┬─────┘         │              │
           │     │               │              │
           │     ▼               │              │
           │┌──────────┐         │              │
           ││Initialize│         │              │
           ││HTS3DModel│         │              │
           │└────┬─────┘         │              │
           │     │               │              │
           │     ▼               │              │
           │┌──────────┐         │              │
           ││Calculate │         │              │
           ││pos_weight│         │              │
           ││for Loss  │         │              │
           │└────┬─────┘         │              │
           │     │               │              │
           │     ▼               │              │
           │┌──────────┐         │              │
           ││Initialize│         │              │
           ││AdamW     │         │              │
           ││Optimizer │         │              │
           │└────┬─────┘         │              │
           │     │               │              │
           │     ▼               │              │
           │┌──────────┐         │              │
           ││For Each  │◄────────┐              │
           ││Epoch:    │         │              │
           ││max 6     │         │              │
           │└───┬────┬─┘         │              │
           │    │    │           │              │
           │┌───┘    └─────┐     │              │
           ││ Next         │ Done│              │
           │▼              ▼     │              │
           │┌──────────┐ ┌──────────┐          │
           ││Train One │ │Store Fold│          │
           ││Epoch     │ │Predictions│         │
           │└────┬─────┘ └────┬─────┘          │
           │     │            │                │
           │     ▼            │                │
           │┌──────────┐      │                │
           ││Evaluate  │      │                │
           ││Model     │      │                │
           │└────┬─────┘      │                │
           │     │            │                │
           │     ▼            │                │
           │┌──────────┐      │                │
           ││Collect   │      │                │
           ││Metrics & │      │                │
           ││Memory    │      │                │
           ││Usage     │      │                │
           │└────┬─────┘      │                │
           │     │            │                │
           │     ▼            │                │
           │┌──────────┐      │                │
           ││Best AUC? │      │                │
           │└───┬────┬─┘      │                │
           │    │    │        │                │
           │┌───┘    └───┐    │                │
           ││ Yes        │ No │                │
           │▼            ▼    │                │
           │┌──────────┐ ┌──────────┐         │
           ││Save Best │ │Increment │         │
           ││Model     │ │No-Improve│         │
           ││State     │ │Counter   │         │
           │└────┬─────┘ └────┬─────┘         │
           │     │            │               │
           │     ▼            ▼               │
           │┌──────────┐ ┌──────────────┐    │
           ││Early     │ │Early Stop?   │    │
           ││Stop?     │ └───┬───────┬──┘    │
           │└───┬────┬─┘     │       │       │
           │    │    │   ┌───┘       └───┐   │
           │    │    │   │ Yes           │ No│
           │    │    │   ▼               │   │
           │    │    │┌──────────┐       │   │
           │    │    ││Break     │       │   │
           │    │    ││Early     │       │   │
           │    │    ││Stopping  │       │   │
           │    │    │└────┬─────┘       │   │
           │    │    │     │             │   │
           │    │    │     └─────────────┘   │
           │    │    │                       │
           │    └────┘                       │
           │                                 │
           └─────────────────────────────────┘
                           │
                           ▼
                   ┌──────────────┐
                   │Save          │
                   │Predictions   │
                   │CSV with      │
                   │Timestamp     │
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │Save Training │
                   │History JSON  │
                   └──────┬───────┘
                          │
                          ▼
                       ┌──┘
                       │End
                       └──
```

### Feature Generation Flow

```
    ┌──────────────────────┐
    │Start: Feature        │
    │Generation            │
    └──────────┬───────────┘
               │
               ▼
    ┌────────────────────────┐
    │Get SMILES List         │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │Tokenize with ChemBERTa │
    │Tokenizer               │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │Move Token IDs &        │
    │Attention Mask to GPU   │
    └──────────┬─────────────┘
               │
               ▼
        ┌──────────────┐
        │For Each      │◄─────────────────────┐
        │SMILES: 2D    │                      │
        │Features      │                      │
        └───┬───────┬──┘                      │
            │       │                        │
      ┌─────┘       └─────┐                  │
      │ Next              │ Done             │
      ▼                   ▼                  │
┌──────────────┐   ┌──────────────┐         │
│Parse SMILES  │   │Stack to      │         │
│to Molecule   │   │Tensor & Move │         │
└──────┬───────┘   │to GPU        │         │
       │           └──────┬───────┘         │
       ▼                  │                 │
  ┌─────────┐             │                 │
  │Valid?   │             │                 │
  └───┬───┬─┘             │                 │
      │   │               │                 │
  ┌───┘   └───┐           │                 │
  │ No        │ Yes       │                 │
  ▼           ▼           │                 │
┌──────┐ ┌──────────────┐│                 │
│Return│ │Generate      ││                 │
│Zero  │ │Morgan        ││                 │
│Feat  │ │Fingerprint:  ││                 │
│ures  │ │2048 bits     ││                 │
└───┬──┘ └──────┬───────┘│                 │
    │           │        │                 │
    │           ▼        │                 │
    │    ┌──────────────┐│                 │
    │    │Generate      ││                 │
    │    │Physicochemical│                 │
    │    │Features:     ││                 │
    │    │12 dims       ││                 │
    │    └──────┬───────┘│                 │
    │           │        │                 │
    │           ▼        │                 │
    │    ┌──────────────┐│                 │
    │    │Concatenate:  ││                 │
    │    │2060 total    ││                 │
    │    └──────┬───────┘│                 │
    │           │        │                 │
    │           ▼        │                 │
    │    ┌──────────────┐│                 │
    │    │Append to     ││                 │
    │    │Feature List  ││                 │
    │    └──────┬───────┘│                 │
    │           │        │                 │
    │           └────────┘                 │
    │                                     │
    ▼                                     │
┌──────────────┐                         │
│GPU-Accelerated│                         │
│StandardScaler │                         │
└──────┬───────┘                         │
       │                                 │
       ▼                                 │
  ┌──────────────┐                       │
  │For Each      │◄──────────────────────┐
  │SMILES: 3D    │                       │
  │Features      │                       │
  └───┬───────┬──┘                       │
      │       │                         │
  ┌───┘       └─────┐                   │
  │ Next            │ Done              │
  ▼                 ▼                   │
┌──────────────┐ ┌──────────────┐      │
│Parse SMILES  │ │Stack to      │      │
│to Molecule   │ │Tensors & Move│      │
└──────┬───────┘ │to GPU        │      │
       │         └──────┬───────┘      │
       ▼                │              │
  ┌─────────┐           │              │
  │Valid?   │           │              │
  └───┬───┬─┘           │              │
      │   │             │              │
  ┌───┘   └───┐         │              │
  │ No        │ Yes     │              │
  ▼           ▼         │              │
┌──────┐ ┌──────────────┐│              │
│Return│ │Add Hydrogens ││              │
│Zero  │ └──────┬───────┘│              │
│Feat  │        │        │              │
│& Mask│        │        │              │
└───┬──┘        ▼        │              │
    │    ┌──────────────┐│              │
    │    │Embed Molecule││              │
    │    │ETKDGv3       ││              │
    │    └──────┬───────┘│              │
    │           │        │              │
    │           ▼        │              │
    │    ┌──────────────┐│              │
    │    │MMFF Optimize ││              │
    │    └──────┬───────┘│              │
    │           │        │              │
    │           ▼        │              │
    │    ┌──────────────┐│              │
    │    │Compute       ││              │
    │    │Gasteiger     ││              │
    │    │Charges       ││              │
    │    └──────┬───────┘│              │
    │           │        │              │
    │           ▼        │              │
    │    ┌──────────────┐│              │
    │    │Extract Atom  ││              │
    │    │Features:     ││              │
    │    │16 dims per   ││              │
    │    │atom          ││              │
    │    └──────┬───────┘│              │
    │           │        │              │
    │           ▼        │              │
    │    ┌──────────────┐│              │
    │    │Create Atom   ││              │
    │    │Mask          ││              │
    │    └──────┬───────┘│              │
    │           │        │              │
    │           ▼        │              │
    │    ┌──────────────┐│              │
    │    │Append        ││              │
    │    │Features &    ││              │
    │    │Mask          ││              │
    │    └──────┬───────┘│              │
    │           │        │              │
    │           └────────┘              │
    │                                   │
    ▼                                   │
┌──────────────┐                       │
│Return All    │                       │
│Features      │                       │
└──────┬───────┘                       │
       │                               │
       └───────────────────────────────┘
                       │
                       ▼
                    ┌──┘
                    │End
                    └──
```

### Model Forward Pass Flow

```
    ┌──────────────────────┐
    │Start: Forward Pass   │
    └──────────┬───────────┘
               │
               ▼
    ┌────────────────────────┐
    │Input: input_ids,       │
    │attention_mask,         │
    │rdkit_feats,            │
    │atom_features,          │
    │atom_mask               │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │ChemBERTa Encoder:      │
    │Extract Hidden States   │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │Project to EMBED_DIM:   │
    │256                     │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │Ligand3DEncoder:        │
    │Process Atom Features   │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │Concatenate ChemBERTa + │
    │3D Tokens               │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │Concatenate Attention + │
    │Atom Masks              │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │ProteinPocketEncoder:   │
    │Generate Protein Tokens │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │Expand Protein Tokens   │
    │to Batch Size           │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │CrossAttentionFusion:   │
    │Ligand attends to       │
    │Protein                 │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │Masked Pooling:         │
    │Sum / Count             │
    └──────────┬─────────────┘
               │
               ▼
        ┌──────────────┐
        │NaN/Inf in    │
        │lig_pool?     │
        └───┬───────┬──┘
            │       │
      ┌─────┘       └─────┐
      │ Yes               │ No
      ▼                   ▼
┌──────────────┐   ┌──────────────┐
│Replace with  │   │RDKit Branch: │
│Zeros         │   │Process 2D    │
└──────┬───────┘   │Features      │
       │           └──────┬───────┘
       │                  │
       └─────────┬────────┘
                 │
                 ▼
        ┌──────────────┐
        │NaN/Inf in    │
        │rdkit_emb?    │
        └───┬───────┬──┘
            │       │
      ┌─────┘       └─────┐
      │ Yes               │ No
      ▼                   ▼
┌──────────────┐   ┌──────────────┐
│Replace with  │   │Concatenate   │
│Zeros         │   │lig_pool +    │
└──────┬───────┘   │rdkit_emb     │
       │           └──────┬───────┘
       │                  │
       └─────────┬────────┘
                 │
                 ▼
        ┌──────────────┐
        │Classifier:   │
        │Linear Layers │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │Generate      │
        │Logits        │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │NaN/Inf in    │
        │logits?       │
        └───┬───────┬──┘
            │       │
      ┌─────┘       └─────┐
      │ Yes               │ No
      ▼                   ▼
┌──────────────┐   ┌──────────────┐
│Replace with  │   │Return logits,│
│Zeros         │   │attn_weights  │
└──────┬───────┘   └──────┬───────┘
       │                  │
       └─────────┬────────┘
                 │
                 ▼
              ┌──┘
              │End
              └──
```

### Training Loop Flow

```
    ┌──────────────────────┐
    │Start: Training Loop  │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────────┐
    │Initialize: best_auc=-inf,│
    │patience=3, no_improve=0  │
    └──────────┬───────────────┘
               │
               ▼
        ┌──────────────┐
        │For Each CV   │◄────────────────────┐
        │Fold: 3 folds │                     │
        └───┬───────┬──┘                     │
            │       │                       │
      ┌─────┘       └─────┐                 │
      │ Next Fold         │ All Folds       │
      ▼                   ▼                 │
┌──────────┐     ┌──────────────┐          │
│Split     │     │Save Best     │          │
│Train/Val │     │Model,        │          │
└────┬─────┘     │Predictions,  │          │
     │           │History       │          │
     ▼           └──────┬───────┘          │
┌──────────┐            │                  │
│Create New│            │                  │
│HTS3DModel│            │                  │
└────┬─────┘            │                  │
     │                  │                  │
     ▼                  │                  │
┌──────────┐            │                  │
│Calculate │            │                  │
│pos_weight│            │                  │
└────┬─────┘            │                  │
     │                  │                  │
     ▼                  │                  │
┌──────────┐            │                  │
│Initialize│            │                  │
│Optimizer │            │                  │
│& Loss    │            │                  │
└────┬─────┘            │                  │
     │                  │                  │
     ▼                  │                  │
┌──────────┐            │                  │
│For Each  │◄───────────┐                  │
│Epoch:    │            │                  │
│max 6     │            │                  │
└───┬────┬─┘            │                  │
    │    │              │                  │
┌───┘    └─────┐        │                  │
│ Next         │ Done   │                  │
▼              ▼        │                  │
┌──────────┐ ┌──────────┐                 │
│Set Model │ │Store Fold│                 │
│to Train  │ │Predictions│                │
│Mode      │ └────┬─────┘                 │
└────┬─────┘      │                       │
     │            │                       │
     ▼            │                       │
┌──────────┐      │                       │
│For Each  │◄─────┐                       │
│Batch     │      │                       │
└───┬────┬─┘      │                       │
    │    │        │                       │
┌───┘    └─────┐  │                       │
│ Next         │ Done│                    │
│ Done         │     │                    │
▼              ▼     │                    │
┌──────────┐ ┌──────────┐                │
│Move Batch│ │Set Model │                │
│to GPU    │ │to Eval   │                │
└────┬─────┘ │Mode      │                │
     │       └────┬─────┘                │
     ▼            │                      │
┌──────────┐      │                      │
│Zero      │      │                      │
│Gradients │      │                      │
└────┬─────┘      │                      │
     │            │                      │
     ▼            │                      │
┌──────────┐      │                      │
│Forward   │      │                      │
│Pass:     │      │                      │
│Model     │      │                      │
└────┬─────┘      │                      │
     │            │                      │
     ▼            │                      │
┌──────────────┐  │                      │
│NaN/Inf in    │  │                      │
│logits?       │  │                      │
└───┬───────┬──┘  │                      │
    │       │     │                      │
┌───┘       └───┐ │                      │
│ Yes           │ │                      │
│ No            │ │                      │
▼               ▼ │                      │
┌──────────┐ ┌──────────┐              │
│Skip Batch│ │Calculate │              │
└────┬─────┘ │BCE Loss  │              │
     │       └────┬─────┘              │
     │            │                    │
     │            ▼                    │
     │      ┌──────────────┐           │
     │      │NaN/Inf in    │           │
     │      │loss?         │           │
     │      └───┬───────┬──┘           │
     │          │       │              │
     │      ┌───┘       └───┐          │
     │      │ Yes           │ No       │
     │      ▼               ▼          │
     │ ┌──────────┐ ┌──────────────┐  │
     │ │Skip Batch│ │Backward Pass │  │
     │ └────┬─────┘ └──────┬───────┘  │
     │      │              │          │
     │      │              ▼          │
     │      │      ┌──────────────┐   │
     │      │      │NaN/Inf in    │   │
     │      │      │gradients?    │   │
     │      │      └───┬───────┬──┘   │
     │      │          │       │      │
     │      │      ┌───┘       └───┐  │
     │      │      │ Yes           │ No│
     │      │      ▼               ▼  │
     │      │ ┌──────────┐ ┌──────────┐│
     │      │ │Skip Batch│ │Clip      ││
     │      │ └────┬─────┘ │Gradients:││
     │      │      │       │norm=1.0  ││
     │      │      │       └────┬─────┘│
     │      │      │            │      │
     │      │      │            ▼      │
     │      │      │     ┌──────────┐ │
     │      │      │     │Update    │ │
     │      │      │     │Weights   │ │
     │      │      │     └────┬─────┘ │
     │      │      │          │       │
     │      │      │          ▼       │
     │      │      │     ┌──────────┐ │
     │      │      │     │Accumulate│ │
     │      │      │     │Loss      │ │
     │      │      │     └────┬─────┘ │
     │      │      │          │       │
     │      │      │          └───────┘
     │      │      │
     │      └──────┘
     │
     ▼
┌──────────┐
│For Each  │◄────────────────────────────┐
│Val Batch │                             │
└───┬────┬─┘                             │
    │    │                               │
┌───┘    └─────┐                         │
│ Next         │ Done                    │
▼              ▼                         │
┌──────────┐ ┌──────────────┐           │
│Forward   │ │Calculate     │           │
│Pass      │ │Metrics: AUC, │           │
└────┬─────┘ │AP, Precision,│           │
     │       │Recall, F1    │           │
     ▼       └──────┬───────┘           │
┌──────────┐        │                   │
│Apply     │        │                   │
│Sigmoid   │        │                   │
└────┬─────┘        │                   │
     │              │                   │
     ▼              │                   │
┌──────────┐        │                   │
│Collect   │        │                   │
│Predictions│       │                   │
│& Labels  │        │                   │
└────┬─────┘        │                   │
     │              │                   │
     │              │                   │
     ▼              │                   │
┌──────────────┐    │                   │
│Collect Memory│    │                   │
│Usage         │    │                   │
└──────┬───────┘    │                   │
       │            │                   │
       ▼            │                   │
┌──────────────┐    │                   │
│Update Fold   │    │                   │
│History       │    │                   │
└──────┬───────┘    │                   │
       │            │                   │
       ▼            │                   │
  ┌──────────────┐  │                   │
  │AUC >         │  │                   │
  │best_fold_auc?│  │                   │
  └───┬───────┬──┘  │                   │
      │       │     │                   │
  ┌───┘       └───┐ │                   │
  │ Yes           │ No                  │
  ▼               ▼ │                   │
┌──────────┐ ┌──────────┐             │
│Update    │ │Increment │             │
│best_fold │ │no_improve│             │
│_auc &    │ └────┬─────┘             │
│Save State│      │                   │
└────┬─────┘      │                   │
     │            │                   │
     ▼            │                   │
┌──────────────┐  │                   │
│AUC > global  │  │                   │
│best_auc?     │  │                   │
└───┬───────┬──┘  │                   │
    │       │     │                   │
┌───┘       └───┐ │                   │
│ Yes           │ No                  │
▼               │ │                   │
┌──────────┐    │ │                   │
│Save      │    │ │                   │
│Global    │    │ │                   │
│Best State│    │ │                   │
└────┬─────┘    │ │                   │
     │          │ │                   │
     ▼          │ │                   │
┌──────────────┐│ │                   │
│Early Stop?   ││ │                   │
└───┬───────┬──┘│ │                   │
    │       │  │ │                   │
┌───┘       └───┐│ │                  │
│ Yes           ││ │                  │
│ No            ││ │                  │
▼               ││ │                  │
┌──────────┐    ││ │                  │
│Break:    │    ││ │                  │
│Early     │    ││ │                  │
│Stopping  │    ││ │                  │
└────┬─────┘    ││ │                  │
     │          ││ │                  │
     └──────────┘│ │                  │
                 │ │                  │
                 └─┘                  │
                                     │
                                     └─────────────────────┘
```

### Cross-Attention Fusion Flow

```
    ┌──────────────────────┐
    │Start: CrossAttention │
    │Fusion                │
    └──────────┬───────────┘
               │
               ▼
    ┌────────────────────────┐
    │Input: ligand_tokens,   │
    │protein_tokens,         │
    │ligand_mask             │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │Multi-Head Attention:   │
    │ligand queries,         │
    │protein keys/values     │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │Add & LayerNorm:        │
    │residual connection     │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │Feed Forward Network:   │
    │GELU activation         │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │Add & LayerNorm:        │
    │residual connection     │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │Apply Ligand Mask       │
    └──────────┬─────────────┘
               │
               ▼
    ┌────────────────────────┐
    │Return fused tokens,    │
    │attention weights       │
    └──────────┬─────────────┘
               │
               ▼
            ┌──┘
            │End
            └──
```

### Protein Pocket Detection Flow

```
    ┌──────────────────────┐
    │Start: Detect Multiple│
    │Pockets on Protein    │
    └──────────┬───────────┘
               │
               ▼
    ┌────────────────────────┐
    │Input: Residues with    │
    │coordinates (x,y,z)     │
    └──────────┬─────────────┘
               │
               ▼
        ┌──────────────┐
        │Enough        │
        │Residues?     │
        └───┬───────┬──┘
            │       │
      ┌─────┘       └─────┐
      │ No                 │ Yes
      ▼                    ▼
┌──────────────┐   ┌──────────────┐
│Use All as    │   │Compute Pairwise│
│Single Pocket │   │Distances       │
└──────┬───────┘   └──────┬───────┘
       │                  │
       │                  ▼
       │          ┌──────────────┐
       │          │Find Surface  │
       │          │Residues:     │
       │          │Top 40% Most  │
       │          │Exposed       │
       │          └──────┬───────┘
       │                 │
       │                 ▼
       │          ┌──────────────┐
       │          │DBSCAN        │
       │          │Clustering:   │
       │          │Group Residues│
       │          │by Distance   │
       │          └──────┬───────┘
       │                 │
       │                 ▼
       │          ┌──────────────┐
       │          │For Each      │◄──────────────┐
       │          │Cluster       │               │
       │          └───┬───────┬──┘               │
       │              │       │                  │
       │        ┌─────┘       └─────┐            │
       │        │ Next Cluster      │ All Done   │
       │        ▼                   ▼            │
       │  ┌──────────────┐   ┌──────────────┐  │
       │  │Cluster Size  │   │Sort Pockets  │  │
       │  │in Range?     │   │by Chemical   │  │
       │  │(8-25 residues)│   │Diversity      │  │
       │  └───┬───────┬──┘   └──────┬───────┘  │
       │      │       │              │          │
       │  ┌───┘       └───┐          │          │
       │  │ No            │ Yes      │          │
       │  ▼               ▼          │          │
       │┌──────────┐ ┌──────────────┐│          │
       ││Skip      │ │Extract Residue││          │
       ││Cluster   │ │Properties:   ││          │
       │└────┬─────┘ │AA Type,      ││          │
       │     │       │Charge, Hydro ││          │
       │     │       │Coordinates   ││          │
       │     │       └──────┬───────┘│          │
       │     │              │        │          │
       │     │              ▼        │          │
       │     │      ┌──────────────┐ │          │
       │     │      │Add to Pocket │ │          │
       │     │      │List          │ │          │
       │     │      └──────┬───────┘ │          │
       │     │             │         │          │
       │     │             └─────────┘          │
       │     │                                  │
       │     └──────────────────────────────────┘
       │
       ▼
┌──────────────┐
│Select Top N  │
│Pockets       │
│(default: 5)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Return List of│
│Pocket        │
│Templates     │
└──────┬───────┘
       │
       ▼
    ┌──┘
    │End
    └──
```

### Multi-Pocket Encoding Flow

```
    ┌──────────────────────┐
    │Start: Encode Multiple│
    │Pockets               │
    └──────────┬───────────┘
               │
               ▼
    ┌────────────────────────┐
    │Input: List of Pocket   │
    │Templates               │
    └──────────┬─────────────┘
               │
               ▼
        ┌──────────────┐
        │For Each      │◄─────────────────────┐
        │Pocket        │                      │
        │Template      │                      │
        └───┬───────┬──┘                      │
            │       │                        │
      ┌─────┘       └─────┐                  │
      │ Next Pocket        │ All Pockets     │
      ▼                   ▼                  │
┌──────────────┐   ┌──────────────┐         │
│ProteinPocket │   │Stack All     │         │
│Encoder:      │   │Pocket        │         │
│Encode Single │   │Encodings     │         │
│Pocket        │   └──────┬───────┘         │
└──────┬───────┘          │                 │
       │                  │                 │
       ▼                  │                 │
┌──────────────┐          │                 │
│Embed AA Types│          │                 │
│(Embedding)   │          │                 │
└──────┬───────┘          │                 │
       │                  │                 │
       ▼                  │                 │
┌──────────────┐          │                 │
│Project       │          │                 │
│Scalar Feats: │          │                 │
│Charge, Hydro,│          │                 │
│Distance      │          │                 │
└──────┬───────┘          │                 │
       │                  │                 │
       ▼                  │                 │
┌──────────────┐          │                 │
│Project 3D    │          │                 │
│Coordinates   │          │                 │
│(sin encoding)│          │                 │
└──────┬───────┘          │                 │
       │                  │                 │
       ▼                  │                 │
┌──────────────┐          │                 │
│Fuse All      │          │                 │
│Features      │          │                 │
└──────┬───────┘          │                 │
       │                  │                 │
       ▼                  │                 │
┌──────────────┐          │                 │
│Output: Pocket│          │                 │
│Tokens        │          │                 │
│[num_res,     │          │                 │
│ embed_dim]   │          │                 │
└──────┬───────┘          │                 │
       │                  │                 │
       └──────────────────┘                 │
                     │                      │
                     ▼                      │
              ┌──────────────┐              │
              │Average Pool  │              │
              │Each Pocket   │              │
              │to Single     │              │
              │Vector        │              │
              └──────┬───────┘              │
                     │                      │
                     ▼                      │
              ┌──────────────┐              │
              │Multi-Head    │              │
              │Self-Attention:│             │
              │Pockets Attend│              │
              │to Each Other │              │
              └──────┬───────┘              │
                     │                      │
                     ▼                      │
              ┌──────────────┐              │
              │Weighted      │              │
              │Average:      │              │
              │Learnable     │              │
              │Weights per   │              │
              │Pocket        │              │
              └──────┬───────┘              │
                     │                      │
                     ▼                      │
              ┌──────────────┐              │
              │Output:       │              │
              │Combined      │              │
              │Pocket        │              │
              │Representation│              │
              │[embed_dim]   │              │
              └──────┬───────┘              │
                     │                      │
                     ▼                      │
                  ┌──┘                      │
                  │End                      │
                  └─────────────────────────┘
```

### Protein Structure Loading Flow

```
    ┌──────────────────────┐
    │Start: Load Protein   │
    │from PDB File         │
    └──────────┬───────────┘
               │
               ▼
        ┌──────────────┐
        │PDB File      │
        │Provided?     │
        └───┬───────┬──┘
            │       │
      ┌─────┘       └─────┐
      │ No                 │ Yes
      ▼                    ▼
┌──────────────┐   ┌──────────────┐
│Use Default   │   │BioPython     │
│NLRP3 Template│   │Available?    │
└──────┬───────┘   └───┬───────┬──┘
       │               │       │
       │          ┌────┘       └─────┐
       │          │ No               │ Yes
       │          ▼                  ▼
       │   ┌──────────────┐   ┌──────────────┐
       │   │Use Default   │   │Parse PDB     │
       │   │Template      │   │Structure     │
       │   └──────┬───────┘   └──────┬───────┘
       │          │                  │
       │          │                  ▼
       │          │          ┌──────────────┐
       │          │          │Extract       │
       │          │          │Residues:     │
       │          │          │Name & CA     │
       │          │          │Coordinates   │
       │          │          └──────┬───────┘
       │          │                 │
       │          │                 ▼
       │          │          ┌──────────────┐
       │          │          │Convert       │
       │          │          │Coordinates   │
       │          │          │(Angstrom→nm) │
       │          │          └──────┬───────┘
       │          │                 │
       │          │                 ▼
       │          │          ┌──────────────┐
       │          │          │Detect        │
       │          │          │Multiple      │
       │          │          │Pockets       │
       │          │          └──────┬───────┘
       │          │                 │
       │          │                 ▼
       │          │          ┌──────────────┐
       │          │          │Build Pocket  │
       │          │          │Templates     │
       │          │          └──────┬───────┘
       │          │                 │
       │          └─────────────────┘
       │                         │
       └─────────────────────────┘
                           │
                           ▼
                  ┌──────────────┐
                  │Return Pocket │
                  │Templates     │
                  └──────┬───────┘
                         │
                         ▼
                      ┌──┘
                      │End
                      └──
```

---

## Notes

- All flowcharts use text-based ASCII art and can be viewed in any text editor or terminal
- The flowcharts represent the main execution paths; error handling and edge cases are simplified for clarity
- Each script includes extensive error handling and validation that may not be fully represented in the flowcharts
- The ensemble model in HTS.py uses cross-validation with multiple feature selection methods for robust predictions
- HTS_3D.py uses GPU-accelerated feature processing and implements comprehensive NaN/Inf checks throughout training
- HTS_3D.py now supports multi-pocket detection on arbitrary protein structures, testing multiple viable binding pockets simultaneously
