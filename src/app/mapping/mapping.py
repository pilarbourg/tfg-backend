"""
Unified Brain Segmentation Pipeline
1. Extracts Cortical regions from fsaverage (Desikan-Killiany Atlas)
2. Extracts Subcortical regions from aseg.mgz (Marching Cubes)
3. Generates a unified mapping.json for the RAG + React 3Fiber Visualizer

Dependencies: uv pip install nilearn nibabel numpy requests scikit-image
"""

import json
import os
import requests
import numpy as np
import nibabel as nib
from nilearn import datasets
from skimage.measure import marching_cubes

OUTPUT_DIR = "data/brain_regions_NEW"
MAPPING_PATH = os.path.join(OUTPUT_DIR, "mapping_NEW.json")
ANNOT_DIR = os.path.expanduser("~/nilearn_data/fsaverage_annot")
ASEG_PATH = os.path.expanduser("~/mne_data/MNE-fsaverage-data/fsaverage/mri/aseg.mgz")

HEMISPHERES = ["lh", "rh"]
ANNOT_URLS = {
    "lh": "https://raw.githubusercontent.com/nilearn/nilearn/main/nilearn/data/fsaverage/lh.aparc.annot",
    "rh": "https://raw.githubusercontent.com/nilearn/nilearn/main/nilearn/data/fsaverage/rh.aparc.annot",
}

PD_RELEVANT_NAMES = {
    "caudate", "putamen", "pallidum", "accumbens", "ventraldc", 
    "brainstem", "thalamus", "hippocampus", "amygdala", "precentral", 
    "postcentral", "superiorfrontal", "rostralmiddlefrontal",
    "isthmuscingulate", "posteriorcingulate", "insula",
    "superiortemporal", "middletemporal", "inferiorparietal", "supramarginal"
}

SUBCORTICAL_LABELS = {
    10: ("lh", "thalamus"), 11: ("lh", "caudate"), 12: ("lh", "putamen"),
    13: ("lh", "pallidum"), 17: ("lh", "hippocampus"), 18: ("lh", "amygdala"),
    26: ("lh", "accumbens"), 28: ("lh", "ventraldc"), 49: ("rh", "thalamus"),
    50: ("rh", "caudate"), 51: ("rh", "putamen"), 52: ("rh", "pallidum"),
    53: ("rh", "hippocampus"), 54: ("rh", "amygdala"), 58: ("rh", "accumbens"),
    60: ("rh", "ventraldc"), 16: ("mid", "brainstem")
}

def download_fsaverage():
    print("Loading fsaverage GIFTI surfaces via nilearn...")
    fs = datasets.fetch_surf_fsaverage("fsaverage")
    print("Surfaces ready.")
    return fs

def download_annotations():
    os.makedirs(ANNOT_DIR, exist_ok=True)
    annot_paths = {}

    for hemi, url in ANNOT_URLS.items():
        out_path = os.path.join(ANNOT_DIR, f"{hemi}.aparc.annot")
        if os.path.exists(out_path):
            print(f"  Annotation already cached: {out_path}")
            annot_paths[hemi] = out_path
            continue

        print(f"  Downloading {hemi}.aparc.annot ...")
        res = requests.get(url, timeout=30)
        if res.status_code != 200:
            print(f"  GitHub fetch failed ({res.status_code}), trying nilearn fetcher...")
            fs5 = datasets.fetch_surf_fsaverage("fsaverage5")
            nilearn_annot = os.path.join(
                os.path.expanduser("~/nilearn_data"),
                "fsaverage5", f"{hemi}.aparc.annot"
            )
            if os.path.exists(nilearn_annot):
                import shutil
                shutil.copy(nilearn_annot, out_path)
            else:
                raise FileNotFoundError(
                    f"Could not download {hemi}.aparc.annot.\n"
                    f"Please manually download from:\n"
                    f"https://github.com/nilearn/nilearn/tree/main/nilearn/data/fsaverage"
                )
        else:
            with open(out_path, "wb") as f:
                f.write(res.content)
            print(f"  Saved → {out_path}")

        annot_paths[hemi] = out_path

    return annot_paths

def write_obj(filepath, vertices, faces, comment="Mesh"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(f"# {comment}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def process_cortical(hemi, fsaverage, annot_paths):
    hemi_long = "left" if hemi == "lh" else "right"
    img = nib.load(fsaverage[f"pial_{hemi_long}"])
    coords, faces = img.darrays[0].data, img.darrays[1].data
    labels, _, region_names = nib.freesurfer.read_annot(annot_paths[hemi])
    
    region_names = [r.decode("utf-8") if isinstance(r, bytes) else r for r in region_names]
    regions = []

    for idx, name in enumerate(region_names):
        if name in ("unknown", "corpuscallosum", "???"): continue
        
        mask = labels == idx
        if mask.sum() < 10: continue
        
        face_mask = np.all(mask[faces], axis=1)
        region_faces = faces[face_mask]
        if len(region_faces) == 0: continue
        
        unique_verts, inverse = np.unique(region_faces, return_inverse=True)
        safe_name = name.lower().replace("-", "_")
        filename = f"{hemi}_{safe_name}.obj"
        
        write_obj(os.path.join(OUTPUT_DIR, filename), coords[unique_verts], inverse.reshape(-1, 3), f"Cortical: {name}")
        
        regions.append({
            "id": f"{hemi}_{safe_name}", "hemisphere": hemi, "region_name": name,
            "file": filename, "parkinsons_relevant": name.lower() in PD_RELEVANT_NAMES, "source": "cortical"
        })
    return regions

def process_subcortical():
    img = nib.load(ASEG_PATH)
    vol, affine = np.asarray(img.dataobj), img.affine
    regions = []

    for label_id, (hemi, name) in SUBCORTICAL_LABELS.items():
        mask = (vol == label_id).astype(np.float32)
        if mask.sum() < 100: continue
        
        verts_vox, faces, _, _ = marching_cubes(mask, level=0.5)
        verts_mm = (affine @ np.hstack([verts_vox, np.ones((len(verts_vox), 1))]).T).T[:, :3]
        
        filename = f"{hemi}_{name}.obj"
        write_obj(os.path.join(OUTPUT_DIR, filename), verts_mm, faces, f"Subcortical: {name}")
        
        regions.append({
            "id": f"{hemi}_{name}", "hemisphere": hemi, "region_name": name,
            "file": filename, "parkinsons_relevant": name in PD_RELEVANT_NAMES, "source": "subcortical"
        })
    return regions

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Prepare Data Sources
    fs_data = download_fsaverage()
    annot_paths = download_annotations()
    
    all_data = []
    
    # 2. Extract Cortical (Loop through hemispheres)
    print("Extracting Cortical Layers...")
    for hemi in HEMISPHERES:
        # Pass the required arguments here!
        hemi_regions = process_cortical(hemi, fs_data, annot_paths)
        all_data.extend(hemi_regions)
    
    # 3. Extract Subcortical
    print("\nExtracting Subcortical Structures...")
    if os.path.exists(ASEG_PATH):
        sub_regions = process_subcortical()
        all_data.extend(sub_regions)
    else:
        print(f"⚠ Skipping Subcortical: {ASEG_PATH} not found.")
        print("Tip: Run 'python -c \"import mne; mne.datasets.fetch_fsaverage()\"' first.")
    
    # 4. Generate Unified Mapping
    # Using 'id' as key to ensure uniqueness
    mapping = {r["id"]: {**r, "metabolites": []} for r in all_data}
    
    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    
    total = len(all_data)
    pk_count = sum(1 for r in all_data if r["parkinsons_relevant"])
    print(f"\n{'─'*55}")
    print(f"Success! {total} total OBJ files written.")
    print(f"{pk_count} regions flagged as Parkinson's relevant.")
    print(f"Mapping saved to: {MAPPING_PATH}")
    print(f"{'─'*55}")