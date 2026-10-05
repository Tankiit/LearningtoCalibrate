"""Local Communities/ACS adapters; downloads are explicit and never implicit."""
from pathlib import Path
import csv
import hashlib
import numpy as np


def load_communities(path):
    p=Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Provide UCI communities.data at {p}; see data/README.md")
    raw=np.asarray(list(csv.reader(p.open())))
    if raw.ndim!=2 or raw.shape[1]!=128:
        raise ValueError("Expected the original 128-column UCI communities.data schema")
    numeric=np.where(raw[:,5:]=="?", "nan", raw[:,5:]).astype(float)
    X,y=numeric[:,:-1],numeric[:,-1]
    if not np.isfinite(y).all():
        raise ValueError("Missing/nonfinite targets are not silently discarded")
    # Explicit given-group comparator: state code, not a learned sensitive-group rule.
    return X,y,raw[:,0],None,{"dataset":"communities", "task":"regression",
        "given_group":"state", "sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
        "source":"https://archive.ics.uci.edu/dataset/183/communities+and+crime"}


def load_acs(root, state="CA", year="2018", download=False):
    from folktables import ACSDataSource, ACSIncome
    from folktables.acs import adult_filter
    source=ACSDataSource(survey_year=year,horizon="1-Year",survey="person",root_dir=str(root))
    frame=source.get_data(states=[state],download=download)
    # Apply the official income filter once, retaining household ids for splitting.
    frame=adult_filter(frame)
    # One label-independent person per household, not dependent row calibration.
    original_rows=len(frame)
    frame=frame.groupby("SERIALNO",sort=True,group_keys=False).sample(n=1,random_state=0)
    X=frame[ACSIncome.features].to_numpy(dtype=float)
    y=(frame["PINCP"].to_numpy()>50000).astype(int)
    groups=frame["RAC1P"].astype(str).to_numpy()
    households=frame["SERIALNO"].astype(str).to_numpy()
    from mondrian.audit import array_hash
    return X,y,groups,households,{"dataset":"acs_income", "task":"classification",
        "given_group":"RAC1P", "state":state, "year":year,
        "sha256":array_hash(np.column_stack([X,y])), "split_unit":"household",
        "sampling":"one eligible person per household, random_state=0",
        "original_eligible_rows":original_rows,"used_households":len(frame),
        "source":"https://github.com/socialfoundations/folktables"}


def load(dataset="communities", path="data/raw/communities.data", **kwargs):
    if dataset=="communities":
        return load_communities(path)
    if dataset=="acs_income":
        return load_acs(path,**kwargs)
    raise ValueError(dataset)
