import os
import json
from pathlib import Path
from tqdm import tqdm
import yaml
import pandas as pd

import cv2
from pytorch_lightning import Trainer
from models.CountingAnything import CountingAnything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger


def validate_checkpoint(cfg, version):
    # given a single checkpoint path, update the results json
    print("test_split", cfg["test_split"])
    print("resume_path", cfg["resume_path"])
    model = CountingAnything(cfg)

    log_folder = f"{cfg['name']}_{cfg['test_split']}"
    logger = TensorBoardLogger(
        cfg["log_dir"],
        name=log_folder,
        version=version,
        default_hp_metric=False,
    )

    val_trainer = Trainer(
        gpus=-1,
        logger=logger,
        max_epochs=cfg["max_epochs"],
        max_steps=cfg["max_steps"],
        accelerator="gpu",
        strategy=DDPStrategy(),
        overfit_batches=cfg["overfit_batches"],
        check_val_every_n_epoch=cfg["val_every"],
        log_every_n_steps=1,
    )

    if cfg["test_split"] == "val":
        val_res = "results_val.json"
    else:
        val_res = "results_test.json"

    if not os.path.exists(val_res):
        data = {"": ""}
        with open(val_res, "w") as outfile:
            json.dump(data, outfile)

    val_results = val_trainer.validate(model)
    val_results = val_results[0]

    val_results.update(cfg)

    if not cfg["save_preds"]:
        assert len(val_trainer.val_dataloaders) == 1, "There should be only one validation dataloader"
        model.eval().to('cuda')

        res = []
        for batch in val_trainer.val_dataloaders[0]:
            inps, gts, im_ids = batch[0], batch[3], batch[4]
            output = model(inps.to('cuda'))

            for i, o, g in zip(output.flatten().cpu().tolist(), gts.flatten().cpu().tolist(), im_ids):
                res.append([i, o, g])

        res_df = pd.DataFrame(res, columns=["pred", "gt", "im_id"])
        res_df.to_csv(os.path.join(cfg["log_dir"], log_folder, version, "results.csv"), index=False)
        with open(val_res, "r") as f:
            dic = json.load(f)

        dic[cfg["resume_path"]] = val_results

        with open(val_res, "w") as f:
            json.dump(dic, f, indent=4, sort_keys=True)
    
    return val_results


def report_all_results(log_dir: str, default_cfg: dict, save_preds: bool = False) -> pd.DataFrame:
    """
    Report all results from checkpoints in the log directory.

    Args:
        log_dir (str): Directory containing the JSON files with results.
        default_cfg (dict): Default configuration for the model.
        save_preds (bool): Whether to save images or not.
    
    Returns:
        res_df (pd.DataFrame): DataFrame containing the results.
    """
    res = []
    log_dir = Path(log_dir)
    for version in log_dir.iterdir():
        version_dir = log_dir / version
        checkpoints_dir = version_dir / 'checkpoints'

        last_epoch = 0
        best_checkpoint = None
        best_mae = float('inf')
        for checkpoint in checkpoints_dir.iterdir():
            if checkpoint.name == 'last.ckpt':
                continue
            cur_epoch = int(checkpoint.name.split('_')[0].split('epoch=')[-1])
            cur_mae = float(checkpoint.name.split('DDP_MAE=')[1].split('_val')[0])
            if cur_mae < best_mae or (cur_mae == best_mae and cur_epoch > last_epoch):
                best_mae = cur_mae
                last_epoch = cur_epoch
                best_checkpoint = checkpoint

        training_config = yaml.safe_load(open(version_dir / f"ykkap_training.yml"))

        training_config["resume_path"] = str(best_checkpoint)
        training_config["name"] = "report"
        training_config["test"] = True
        training_config["test_split"] = "test"
        training_config["save_preds"] = save_preds
        training_config["rounding_count"] = True
        training_config = {**default_cfg, **training_config}
        val_res = validate_checkpoint(training_config, version.name)
        res.append({
            'version': version.name.replace('version_', ''),
            'backbone': training_config['backbone'],
            'backbone_unfreeze_layers': training_config['backbone_unfreeze_layers'],
            'counting_head': training_config['counting_head'],
            'learning_rate': training_config['learning_rate'],
            'counting_loss': training_config['counting_loss'],
            'train_batch_size': training_config['train_batch_size'],
            'test_mae': round(val_res['test_DDP_MAE'], 2),
        })

    res_df = pd.DataFrame(res)
    return res_df


def visualize_predictions(log_dir: str, cfg: dict):
    """
    Visualize predictions from the model.

    Args:
        log_dir (str): Directory containing the JSON files with results.
        cfg (dict): Configuration for the model.
    """
    assert cfg["test_split"] in ["val", "test"], "test_split should be val or test"
    log_dir = Path(log_dir)

    results_path = log_dir / "results.csv"
    img_dir = Path(f"{cfg['data_path']}/{cfg['dataset']}/images")
    assert results_path.exists(), f"results.csv does not exist in {log_dir}"

    res_df = pd.read_csv(results_path)
    for _, row in tqdm(res_df.iterrows()):
        im_id = row["im_id"]
        pred = int(row["pred"])
        gt = int(row["gt"])

        img_path = img_dir / im_id
        assert img_path.exists(), f"Image {img_path} does not exist"

        # Load and visualize the image and predictions
        img = cv2.imread(str(img_path))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        pred_pos = (10, 20)
        gt_pos = (10, 60)
        cv2.putText(img, f"Pred: {pred}", pred_pos, font, font_scale, (255, 0, 0), thickness)
        cv2.putText(img, f"GT: {gt}", gt_pos, font, font_scale, (100, 200, 100), thickness)
        
        output_path = log_dir / "output" / im_id
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)

