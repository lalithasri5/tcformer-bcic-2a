import os
import time
import yaml
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from utils.plotting import plot_confusion_matrix, plot_curve
from utils.metrics import MetricsCallback, write_summary
from utils.latency import measure_latency
from utils.misc import visualize_model_graph, show_gpu_info

from utils.get_datamodule_cls import get_datamodule_cls
from utils.get_model_cls import get_model_cls
from utils.seed import seed_everything


# ============================================================
# CONFIG DIRECTORY
# ============================================================

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


# ============================================================
# MAIN TRAINING + TESTING
# ============================================================

def train_and_test(config):

    # --------------------------------------------------------
    # RESULT DIRECTORY
    # --------------------------------------------------------

    model_name = config["model"]
    dataset_name = config["dataset_name"]
    seed = config["seed"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    result_dir = (
        Path(__file__).resolve().parent /
        f"results/{model_name}_{dataset_name}"
        f"_seed-{seed}"
        f"_aug-{config['preprocessing']['interaug']}"
        f"_GPU{config['gpu_id']}_{timestamp}"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for sub in [
        "checkpoints",
        "confmats",
        "curves"
    ]:
        (result_dir / sub).mkdir(
            parents=True,
            exist_ok=True
        )


    # --------------------------------------------------------
    # SAVE CONFIG
    # --------------------------------------------------------

    with open(
        result_dir / "config.yaml",
        "w"
    ) as f:

        yaml.dump(
            config,
            f,
            default_flow_style=False
        )


    # --------------------------------------------------------
    # MODEL + DATAMODULE
    # --------------------------------------------------------

    model_cls = get_model_cls(model_name)

    datamodule_cls = get_datamodule_cls(
        dataset_name
    )

    config["model_kwargs"]["n_channels"] = (
        datamodule_cls.channels
    )

    config["model_kwargs"]["n_classes"] = (
        datamodule_cls.classes
    )


    # --------------------------------------------------------
    # SUBJECT IDs
    # --------------------------------------------------------

    subj_cfg = config["subject_ids"]

    subject_ids = (
        datamodule_cls.all_subject_ids
        if subj_cfg == "all"
        else [subj_cfg]
        if isinstance(subj_cfg, int)
        else subj_cfg
    )


    # --------------------------------------------------------
    # METRIC CONTAINERS
    # --------------------------------------------------------

    test_accs = []
    test_losses = []
    test_kappas = []

    train_times = []
    test_times = []
    response_times = []

    all_confmats = []


    # ========================================================
    # SUBJECT LOOP
    # ========================================================

    for subject_id in subject_ids:

        print(
            f"\n>>> Training on subject: {subject_id}"
        )


        # ----------------------------------------------------
        # SEED
        # ----------------------------------------------------

        seed_everything(
            config["seed"]
        )


        # ----------------------------------------------------
        # METRICS CALLBACK
        # ----------------------------------------------------

        metrics_callback = MetricsCallback()


        # ----------------------------------------------------
        # TRAINER
        #
        # IMPORTANT:
        # limit_val_batches=0 has been REMOVED.
        #
        # Validation is now enabled.
        # ----------------------------------------------------

        trainer = Trainer(

            max_epochs=config["max_epochs"],

            devices=(
                -1
                if config.get("gpu_id", 0) == -1
                else [config.get("gpu_id", 0)]
            ),

            num_sanity_val_steps=0,

            accelerator="auto",

            strategy=(
                "auto"
                if config.get("gpu_id", 0) != -1
                else DDPStrategy(
                    find_unused_parameters=True
                )
            ),

            logger=False,

            enable_checkpointing=False,

            callbacks=[
                metrics_callback
            ],

            log_every_n_steps=1,

            enable_progress_bar=True
        )


        # ----------------------------------------------------
        # DATAMODULE
        # ----------------------------------------------------

        datamodule = datamodule_cls(
            config["preprocessing"],
            subject_id=subject_id
        )


        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = model_cls(
            **config["model_kwargs"],
            max_epochs=config["max_epochs"]
        )


        # ----------------------------------------------------
        # PARAMETER COUNT
        # ----------------------------------------------------

        param_count = sum(
            p.numel()
            for p in model.parameters()
        )


        # ====================================================
        # TRAIN
        # ====================================================

        print("=" * 70)
        print(
            f"START TRAINING SUBJECT {subject_id}"
        )
        print("=" * 70)

        st_train = time.time()

        print(
            ">>> BEFORE trainer.fit()"
        )

        trainer.fit(
            model,
            datamodule=datamodule
        )

        print(
            ">>> AFTER trainer.fit()"
        )

        print("=" * 70)
        print(
            f"FINISHED TRAINING SUBJECT {subject_id}"
        )
        print("=" * 70)


        train_times.append(
            (time.time() - st_train) / 60
        )


        # ====================================================
        # SAVE TRAINING HISTORY
        # ====================================================

        print("\nSaving training history...")


        train_acc = np.array(
            metrics_callback.train_acc,
            dtype=float
        )

        val_acc = np.array(
            metrics_callback.val_acc,
            dtype=float
        )

        train_loss = np.array(
            metrics_callback.train_loss,
            dtype=float
        )

        val_loss = np.array(
            metrics_callback.val_loss,
            dtype=float
        )


        print(
            "Training accuracy epochs:",
            len(train_acc)
        )

        print(
            "Validation accuracy epochs:",
            len(val_acc)
        )


        # ----------------------------------------------------
        # Make sure both curves have matching epochs
        # ----------------------------------------------------

        n_epochs = min(
            len(train_acc),
            len(val_acc)
        )


        # ----------------------------------------------------
        # History DataFrame
        # ----------------------------------------------------

        history = pd.DataFrame({

            "epoch": np.arange(
                1,
                n_epochs + 1
            ),

            "train_accuracy": (
                train_acc[:n_epochs]
            ),

            "validation_accuracy": (
                val_acc[:n_epochs]
            ),

            "train_loss": (
                train_loss[:n_epochs]
                if len(train_loss) >= n_epochs
                else np.nan
            ),

            "validation_loss": (
                val_loss[:n_epochs]
                if len(val_loss) >= n_epochs
                else np.nan
            )
        })


        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        history_path = (
            result_dir /
            f"training_history_subject_{subject_id}.csv"
        )

        history.to_csv(
            history_path,
            index=False
        )


        print(
            "Training history saved:"
        )

        print(history_path)


        # ====================================================
        # FIGURE 1
        # TRAINING VS VALIDATION ACCURACY
        # ====================================================

        if n_epochs > 0:

            plt.figure(
                figsize=(10, 6)
            )

            plt.plot(
                history["epoch"],
                history["train_accuracy"] * 100,
                label="Training Accuracy",
                linewidth=2
            )

            plt.plot(
                history["epoch"],
                history["validation_accuracy"] * 100,
                label="Validation Accuracy",
                linewidth=2
            )

            plt.xlabel(
                "Epoch",
                fontsize=12
            )

            plt.ylabel(
                "Accuracy (%)",
                fontsize=12
            )

            plt.title(
                f"Training vs Validation Accuracy — Subject {subject_id}",
                fontsize=14
            )

            plt.legend()

            plt.grid(
                alpha=0.2
            )

            plt.tight_layout()


            train_val_path = (
                result_dir /
                f"training_vs_validation_subject_{subject_id}.png"
            )

            plt.savefig(
                train_val_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()


            print(
                "Training vs validation figure saved:"
            )

            print(train_val_path)


        # ====================================================
        # FIGURE 2
        # EPOCH VS ACCURACY
        # ====================================================

        if n_epochs > 0:

            plt.figure(
                figsize=(10, 6)
            )

            plt.plot(
                history["epoch"],
                history["train_accuracy"] * 100,
                label="Training Accuracy",
                linewidth=2
            )

            plt.xlabel(
                "Epoch",
                fontsize=12
            )

            plt.ylabel(
                "Accuracy (%)",
                fontsize=12
            )

            plt.title(
                f"Epoch vs Training Accuracy — Subject {subject_id}",
                fontsize=14
            )

            plt.legend()

            plt.grid(
                alpha=0.2
            )

            plt.tight_layout()


            epoch_accuracy_path = (
                result_dir /
                f"epoch_vs_accuracy_subject_{subject_id}.png"
            )

            plt.savefig(
                epoch_accuracy_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()


            print(
                "Epoch vs accuracy figure saved:"
            )

            print(epoch_accuracy_path)


        # ====================================================
        # OPTIONAL ORIGINAL CURVE
        # ====================================================

        if (
            metrics_callback.train_loss
            and metrics_callback.val_loss
        ):

            plot_curve(
                metrics_callback.train_loss,
                metrics_callback.val_loss,
                "Loss",
                subject_id,
                result_dir /
                f"curves/subject_{subject_id}_loss.png"
            )


        if (
            metrics_callback.train_acc
            and metrics_callback.val_acc
        ):

            plot_curve(
                metrics_callback.train_acc,
                metrics_callback.val_acc,
                "Accuracy",
                subject_id,
                result_dir /
                f"curves/subject_{subject_id}_acc.png"
            )


        # ====================================================
        # TEST
        # ====================================================

        print("=" * 70)
        print(
            f"START TESTING SUBJECT {subject_id}"
        )
        print("=" * 70)

        st_test = time.time()

        test_results = trainer.test(
            model,
            datamodule
        )

        print("=" * 70)
        print(
            f"FINISHED TESTING SUBJECT {subject_id}"
        )
        print("=" * 70)


        test_duration = (
            time.time() - st_test
        )

        test_times.append(
            test_duration
        )


        # ====================================================
        # LATENCY
        # ====================================================

        sample_x, _ = (
            datamodule.test_dataset[0]
        )

        input_shape = (
            1,
            *sample_x.shape
        )

        device_str = "cpu"

        lat_ms = measure_latency(
            model,
            input_shape,
            device=device_str
        )

        response_times.append(
            lat_ms
        )


        # ====================================================
        # GPU CLEANUP
        # ====================================================

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            print(
                f"GPU Allocated : "
                f"{torch.cuda.memory_allocated()/1024**2:.2f} MB"
            )

            print(
                f"GPU Reserved  : "
                f"{torch.cuda.memory_reserved()/1024**2:.2f} MB"
            )


        # ====================================================
        # TEST METRICS
        # ====================================================

        test_accs.append(
            test_results[0]["test_acc"]
        )

        test_losses.append(
            test_results[0]["test_loss"]
        )

        test_kappas.append(
            test_results[0]["test_kappa"]
        )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        cm = model.test_confmat.numpy()

        all_confmats.append(
            cm
        )


        if config.get(
            "plot_cm_per_subject",
            False
        ):

            plot_confusion_matrix(

                cm,

                save_path=(
                    result_dir /
                    f"confmats/"
                    f"confmat_subject_{subject_id}.png"
                ),

                class_names=(
                    datamodule_cls.class_names
                ),

                title=(
                    f"Confusion Matrix – "
                    f"Subject {subject_id}"
                )
            )


        # ====================================================
        # SAVE CHECKPOINT
        # ====================================================

        if config.get(
            "save_checkpoint",
            False
        ):

            ckpt_path = (
                result_dir /
                f"checkpoints/"
                f"subject_{subject_id}_model.ckpt"
            )

            trainer.save_checkpoint(
                ckpt_path
            )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    write_summary(

        result_dir,
        model_name,
        dataset_name,
        subject_ids,
        param_count,
        test_accs,
        test_losses,
        test_kappas,
        train_times,
        test_times,
        response_times
    )


    # ========================================================
    # AVERAGE CONFUSION MATRIX
    # ========================================================

    if (
        config.get(
            "plot_cm_average",
            True
        )
        and all_confmats
    ):

        avg_cm = np.mean(
            np.stack(all_confmats),
            axis=0
        )

        plot_confusion_matrix(

            avg_cm,

            save_path=(
                result_dir /
                "confmats/"
                "avg_confusion_matrix.png"
            ),

            class_names=(
                datamodule_cls.class_names
            ),

            title="Average Confusion Matrix"
        )


# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_arguments():

    parser = ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="tcformer",
        help=(
            "Name of the model to use. Options:\n"
            "tcformer, atcnet, d-atcnet, "
            "atcnet_2_0, eegnet, shallownet, "
            "basenet, eegtcnet, eegconformer, "
            "tsseffnet, eegdeformer, sst_dpn, "
            "ctnet, mscformer"
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="bcic2a",
        help=(
            "Name of the dataset to use. "
            "Options: bcic2a, bcic2b, hgd, "
            "reh_mi, bcic3"
        )
    )

    parser.add_argument(
        "--loso",
        action="store_true",
        default=False,
        help="Enable subject-independent (LOSO) mode"
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU device ID to use"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed value"
    )

    parser.add_argument(
        "--interaug",
        action="store_true",
        help=(
            "Enable inter-trial augmentation"
        )
    )

    parser.add_argument(
        "--no_interaug",
        action="store_true",
        help=(
            "Disable inter-trial augmentation"
        )
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def run():

    args = parse_arguments()

    config_path = os.path.join(
        CONFIG_DIR,
        f"{args.model}.yaml"
    )

    with open(config_path) as f:

        config = yaml.safe_load(f)


    # --------------------------------------------------------
    # LOSO
    # --------------------------------------------------------

    if args.loso:

        config["dataset_name"] = (
            args.dataset + "_loso"
        )

        config["max_epochs"] = (

            config["max_epochs_loso_hgd"]

            if args.dataset == "hgd"

            else config["max_epochs_loso"]
        )

        config["model_kwargs"][
            "warmup_epochs"
        ] = (
            config["model_kwargs"][
                "warmup_epochs_loso"
            ]
        )


    # --------------------------------------------------------
    # NORMAL TRAINING
    # --------------------------------------------------------

    else:

        if args.dataset == "bcic2a":

            config["dataset_name"] = (
                "bcic2a_tvt"
            )

        else:

            config["dataset_name"] = (
                args.dataset
            )


        config["max_epochs"] = (

            config["max_epochs_2b"]

            if args.dataset == "bcic2b"

            else config["max_epochs"]
        )


    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    config["preprocessing"] = (
        config["preprocessing"][
            args.dataset
        ]
    )

    config["preprocessing"][
        "z_scale"
    ] = config["z_scale"]


    # --------------------------------------------------------
    # INTERAUG
    # --------------------------------------------------------

    if args.interaug:

        config["preprocessing"][
            "interaug"
        ] = True

    elif args.no_interaug:

        config["preprocessing"][
            "interaug"
        ] = False

    else:

        config["preprocessing"][
            "interaug"
        ] = config["interaug"]


    config.pop(
        "interaug",
        None
    )


    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    config["gpu_id"] = args.gpu_id


    # --------------------------------------------------------
    # SEED
    # --------------------------------------------------------

    if args.seed is not None:

        config["seed"] = args.seed


    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    config["plot_cm_per_subject"] = True

    config["plot_cm_average"] = True


    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    train_and_test(config)


if __name__ == "__main__":

    run()