import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

def event2dataframe(event_path: str)
    ea = event_accumulator.EventAccumulator(
        event_path,
        size_guidance={
            event_accumulator.SCALARS: 0,
            event_accumulator.TENSORS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.HISTOGRAMS: 0,
        },
    ).Reload()
    df = pd.DataFrame(columns=ea.Tags()["scalars"])
    df.index.name = "step"
    pbar = tqdm(ea.Tags()["scalars"], desc="Loading scalar tags")
    for scalar in pbar:
        pbar.set_postfix_str(f"tag={scalar}")
        for event in ea.Scalars(scalar):
            df.loc[event.step, scalar] = event.value
    return df