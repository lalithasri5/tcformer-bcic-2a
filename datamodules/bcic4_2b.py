from typing import Optional

import numpy as np
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split

from .base import BaseDataModule
from utils.load_bcic4 import load_bcic4


class BCICIV2b(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["hand(L)", "hand(R)"]
    channels = 3
    classes = 2

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(
            subject_ids=[int(self.subject_id)],
            dataset="2b",
            preprocessing_dict=self.preprocessing_dict
        )

    def setup(self, stage: Optional[str] = None) -> None:

        if self.dataset is None:
            self.prepare_data()

        # ====================================================
        # SPLIT DATA BY SESSION
        # ====================================================

        splitted_ds = self.dataset.split("session")

        # Training sessions
        train_datasets = [
            splitted_ds["0train"],
            splitted_ds["1train"],
            splitted_ds["2train"],
        ]

        # Test sessions - NEVER used for validation
        test_datasets = [
            splitted_ds["3test"],
            splitted_ds["4test"],
        ]

        # ====================================================
        # LOAD TRAINING DATA
        # ====================================================

        X = np.concatenate(
            [
                np.stack(
                    [run[i][0] for i in range(len(run))]
                )
                for train_dataset in train_datasets
                for run in train_dataset.datasets
            ],
            axis=0
        )

        y = np.concatenate(
            [
                run.y
                for train_dataset in train_datasets
                for run in train_dataset.datasets
            ],
            axis=0
        )

        # ====================================================
        # LOAD TEST DATA
        # ====================================================

        X_test = np.concatenate(
            [
                np.stack(
                    [run[i][0] for i in range(len(run))]
                )
                for test_dataset in test_datasets
                for run in test_dataset.datasets
            ],
            axis=0
        )

        y_test = np.concatenate(
            [
                run.y
                for test_dataset in test_datasets
                for run in test_dataset.datasets
            ],
            axis=0
        )

        # ====================================================
        # 80% TRAIN / 20% VALIDATION
        #
        # ONLY the original training sessions are split.
        # Test sessions remain completely untouched.
        # ====================================================

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            shuffle=True,
            stratify=y
        )

        # ====================================================
        # Z-SCORE
        #
        # StandardScaler is fitted ONLY on training data.
        # Validation and test are transformed using the
        # training statistics.
        # ====================================================

        if self.preprocessing_dict["z_scale"]:

            X_train, X_val, X_test = BaseDataModule._z_scale_tvt(
                X_train,
                X_val,
                X_test
            )

        # ====================================================
        # CREATE PYTORCH DATASETS
        # ====================================================

        self.train_dataset = BaseDataModule._make_tensor_dataset(
            X_train,
            y_train
        )

        self.val_dataset = BaseDataModule._make_tensor_dataset(
            X_val,
            y_val
        )

        self.test_dataset = BaseDataModule._make_tensor_dataset(
            X_test,
            y_test
        )

        # ====================================================
        # PRINT DATASET INFORMATION
        # ====================================================

        print("\n" + "=" * 60)
        print("BCIC IV-2B DATA SPLIT")
        print("=" * 60)

        print(
            f"Training samples   : {len(self.train_dataset)}"
        )

        print(
            f"Validation samples : {len(self.val_dataset)}"
        )

        print(
            f"Test samples       : {len(self.test_dataset)}"
        )

        print(
            f"Training classes   : {np.bincount(y_train.astype(int))}"
        )

        print(
            f"Validation classes : {np.bincount(y_val.astype(int))}"
        )

        print(
            f"Test classes       : {np.bincount(y_test.astype(int))}"
        )

        print("=" * 60 + "\n")

    # ========================================================
    # VALIDATION DATALOADER
    # ========================================================

    def val_dataloader(self) -> DataLoader:

        return DataLoader(
            self.val_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )


# ============================================================
# LOSO VERSION
# ============================================================

class BCICIV2bLOSO(BCICIV2b):

    val_dataset = None

    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super().__init__(
            preprocessing_dict,
            subject_id
        )

    def prepare_data(self) -> None:

        self.dataset = load_bcic4(
            subject_ids=[
                int(s)
                for s in self.all_subject_ids
            ],
            dataset="2b",
            preprocessing_dict=self.preprocessing_dict
        )

    def setup(self, stage: Optional[str] = None) -> None:

        if self.dataset is None:
            self.prepare_data()

        splitted_ds = self.dataset.split("subject")

        train_subjects = [
            subj_id
            for subj_id in self.all_subject_ids
            if subj_id != self.subject_id
        ]

        # ====================================================
        # TRAIN DATA
        # ====================================================

        train_datasets = [
            splitted_ds[str(subj_id)]
            .split("session")[f"{session}train"]
            for subj_id in train_subjects
            for session in [0, 1, 2]
        ]

        # ====================================================
        # VALIDATION DATA
        # ====================================================

        val_datasets = [
            splitted_ds[str(subj_id)]
            .split("session")[f"{session}test"]
            for subj_id in train_subjects
            for session in [3, 4]
        ]

        # ====================================================
        # TEST DATA
        # ====================================================

        test_datasets = [
            splitted_ds[str(self.subject_id)]
            .split("session")[f"{session}test"]
            for session in [3, 4]
        ]

        # ====================================================
        # LOAD TRAIN
        # ====================================================

        X = np.concatenate(
            [
                np.stack(
                    [run[i][0] for i in range(len(run))]
                )
                for train_dataset in train_datasets
                for run in train_dataset.datasets
            ],
            axis=0
        )

        y = np.concatenate(
            [
                run.y
                for train_dataset in train_datasets
                for run in train_dataset.datasets
            ],
            axis=0
        )

        # ====================================================
        # LOAD VALIDATION
        # ====================================================

        X_val = np.concatenate(
            [
                np.stack(
                    [run[i][0] for i in range(len(run))]
                )
                for val_dataset in val_datasets
                for run in val_dataset.datasets
            ],
            axis=0
        )

        y_val = np.concatenate(
            [
                run.y
                for val_dataset in val_datasets
                for run in val_dataset.datasets
            ],
            axis=0
        )

        # ====================================================
        # LOAD TEST
        # ====================================================

        X_test = np.concatenate(
            [
                np.stack(
                    [run[i][0] for i in range(len(run))]
                )
                for test_dataset in test_datasets
                for run in test_dataset.datasets
            ],
            axis=0
        )

        y_test = np.concatenate(
            [
                run.y
                for test_dataset in test_datasets
                for run in test_dataset.datasets
            ],
            axis=0
        )

        # ====================================================
        # Z-SCORE
        # ====================================================

        if self.preprocessing_dict["z_scale"]:

            X, X_val, X_test = BaseDataModule._z_scale_tvt(
                X,
                X_val,
                X_test
            )

        # ====================================================
        # CREATE DATASETS
        # ====================================================

        self.train_dataset = BaseDataModule._make_tensor_dataset(
            X,
            y
        )

        self.val_dataset = BaseDataModule._make_tensor_dataset(
            X_val,
            y_val
        )

        self.test_dataset = BaseDataModule._make_tensor_dataset(
            X_test,
            y_test
        )

    # ========================================================
    # VALIDATION DATALOADER
    # ========================================================

    def val_dataloader(self) -> DataLoader:

        return DataLoader(
            self.val_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            shuffle=False,
            num_workers=0
        )