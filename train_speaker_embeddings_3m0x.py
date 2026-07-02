#!/usr/bin/python3

import csv
import os
import random
import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio import audio_io


class SpeakerBrain(sb.core.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        feats = self.modules.wav2vec2(wavs)
        embeddings = self.modules.pooling(feats, lens)

        if len(embeddings.shape) == 3:
            embeddings = embeddings.squeeze(1)

        outputs = self.modules.classifier(embeddings)
        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded

        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )


def make_speechbrain_csv(input_csv, output_csv, data_folder, sample_rate):
    """Convert Yousef/3M0X CSV format to SpeechBrain format.

    Input columns:
        profile_id,file,duration,...

    Output columns:
        ID,wav,start,stop,duration,spk_id
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    data_folder = Path(data_folder)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    missing = []

    with input_csv.open("r", encoding="utf-8") as fin, output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(
            fout,
            fieldnames=["ID", "wav", "start", "stop", "duration", "spk_id"],
        )
        writer.writeheader()

        for row in reader:
            wav_name = row["file"]
            profile_id = row["profile_id"]
            duration = float(row["duration"])
            wav_path = data_folder / wav_name

            if not wav_path.exists():
                missing.append(str(wav_path))
                continue

            stop = int(duration * sample_rate)

            writer.writerow(
                {
                    "ID": Path(wav_name).stem,
                    "wav": str(wav_path),
                    "start": 0,
                    "stop": stop,
                    "duration": duration,
                    "spk_id": profile_id,
                }
            )
            rows_written += 1

    if missing:
        print("Missing wav files example:")
        for item in missing[:10]:
            print(item)
        raise RuntimeError(f"Missing {len(missing)} wav files")

    print(f"Created {output_csv} with {rows_written} rows")


def dataio_prep(hparams):
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        start = int(start)
        stop = int(stop)
        duration_sample = stop - start

        if hparams["random_chunk"] and duration_sample > snt_len_sample:
            start = random.randint(0, duration_sample - snt_len_sample)
            stop = start + snt_len_sample

        num_frames = stop - start
        sig, fs = audio_io.load(wav, num_frames=num_frames, frame_offset=start)

        if fs != hparams["sample_rate"]:
            raise RuntimeError(f"Expected {hparams['sample_rate']} Hz, got {fs} Hz for {wav}")

        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="spk_id",
    )

    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data, valid_data, label_encoder


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    os.makedirs(hparams["save_folder"], exist_ok=True)

    make_speechbrain_csv(
        hparams["train_protocol"],
        hparams["train_annotation"],
        hparams["data_folder"],
        hparams["sample_rate"],
    )

    make_speechbrain_csv(
        hparams["valid_protocol"],
        hparams["valid_annotation"],
        hparams["data_folder"],
        hparams["sample_rate"],
    )

    train_data, valid_data, label_encoder = dataio_prep(hparams)

    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
