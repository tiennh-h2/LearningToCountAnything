import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
import argparse
import sys
import yaml
from utils.export_onnx import export_onnx_model
from utils.validation_utils import report_all_results, validate_checkpoint, visualize_predictions

from models.CountingAnything import CountingAnything


def main():
    if CFG["seed"] != -1:
        seed_everything(CFG["seed"], workers=True)

    callbacks = get_callbacks()
    if CFG["test"]:
        validate_checkpoint(CFG)
    else:
        t_logger = TensorBoardLogger(
            CFG["log_dir"], name=CFG["name"], default_hp_metric=False
        )
        trainer = Trainer(
            gpus=-1,
            logger=t_logger,
            max_epochs=CFG["max_epochs"],
            max_steps=CFG["max_steps"],
            accelerator="gpu",
            strategy=DDPStrategy(),
            callbacks=callbacks,
            overfit_batches=CFG["overfit_batches"],
            check_val_every_n_epoch=CFG["val_every"],
            log_every_n_steps=1,
            num_sanity_val_steps=-1,
        )

        model = CountingAnything(CFG)
        trainer.fit(model)


def get_callbacks():
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [lr_monitor]

    if CFG["test_split"] == "val":
        if CFG["use_localisation_head"]:
            model_checkpoint_loc = ModelCheckpoint(
                monitor="val_DDP_Loc_MSE",
                save_last=True,
                save_top_k=3,
                every_n_epochs=1,
                filename="{epoch}_{val_DDP_Loc_MSE:.2f}",
            )
            callbacks.append(model_checkpoint_loc)

        if CFG["use_counting_head"]:
            model_checkpoint_MAE = ModelCheckpoint(
                monitor="val_DDP_MAE",
                save_last=True,
                save_top_k=3,
                every_n_epochs=1,
                filename="{epoch}_{val_DDP_MAE:.2f}_{val_DDP_RMSE:.2f}",
            )
            callbacks.append(model_checkpoint_MAE)
    return callbacks


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Train a 3D reconstruction model.")
    PARSER.add_argument("--config", "-c", type=str, help="Path to config file.")
    PARSER.add_argument(
        "--report", action="store_true", help="Run report for all results in log_dir"
    )
    PARSER.add_argument(
        "--log_dir", type=str, help="Path to log directory.", default=None
    )
    PARSER.add_argument(
        "--test", action="store_true", help="Only run a test script on the test set"
    )
    PARSER.add_argument(
        "--val", action="store_true", help="Only run a test script on the val set"
    )
    PARSER.add_argument(
        "--save_preds", action="store_true", help="save images to visualise"
    )
    PARSER.add_argument(
        "--visualize", action="store_true", help="visualize the predictions"
    )
    PARSER.add_argument(
        "--onnx-export", action="store_true", help="Export the model to ONNX format."
    )
    PARSER.add_argument(
        "--onnx-exported-path", type=str, default="./model.onnx", help="Path to save the exported ONNX model."
    )
    PARSER.add_argument(
        "--model-path", type=str, default=None, help="Path to the model checkpoint."
    )
    PARSER.add_argument(
        "--multi-config-path", type=str, default=None, help="Path to the multi config file."
    )
    ARGS = PARSER.parse_args()
    save_preds = ARGS.save_preds and ARGS.test
    CFG = yaml.safe_load(open("configs/_DEFAULT.yml"))

    if ARGS.report:
        if ARGS.log_dir is None:
            raise ValueError("Please provide a log_dir to report on.")
        res_df = report_all_results(ARGS.log_dir, default_cfg=CFG, save_preds=save_preds)
        print(res_df)
        res_df.to_csv("report.csv", index=False)
        sys.exit(0)

    config_path = ARGS.config if os.path.exists(ARGS.config) else f"configs/{ARGS.config}.yml"
    CFG_new = yaml.safe_load(open(config_path))

    CFG.update(CFG_new)

    CFG["name"] = ARGS.config.split("/")[-1].split(".")[0] if os.path.exists(ARGS.config) else ARGS.config
    CFG["test"] = ARGS.test or ARGS.val

    if ARGS.val:
        CFG["test_split"] = "val"
    elif ARGS.test:
        CFG["test_split"] = "test"

    CFG["save_preds"] = save_preds
    print(CFG) 

    if ARGS.visualize:
        if ARGS.log_dir is None:
            raise ValueError("Please provide a log_dir to visualize.")
        visualize_predictions(ARGS.log_dir, CFG)
        sys.exit(0)


    if ARGS.onnx_export:
        if ARGS.model_path is None:
            raise ValueError("Please provide a model_path to export the ONNX model.")
        export_onnx_model(
            ARGS.model_path,
            CFG,
            ARGS.onnx_exported_path
        )
        sys.exit(0)

    if ARGS.multi_config_path:
        from itertools import product
        from collections import OrderedDict

        multiple_configs = OrderedDict(yaml.safe_load(open(ARGS.multi_config_path)))
        all_keys = multiple_configs.keys()
        all_values = product(*multiple_configs.values())

        for values in all_values:
            for i, key in enumerate(all_keys):
                CFG[key] = values[i]
                CFG["name"] = "_".join([f"{key}_{value}" for key, value in zip(all_keys, values)])
                main()
        sys.exit(0)
    main()
