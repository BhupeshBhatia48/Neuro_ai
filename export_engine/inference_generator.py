"""
export_engine/inference_generator.py
--------------------------------------
Generates a ready-to-run predict.py script tailored to the trained model.

This solves the user-friendliness problem:
  After training, the user has model.pkl and preprocessing.pkl but has
  no idea how to use them. They don't know what columns are needed,
  what values are valid, or how to write the prediction code.

This module generates:
  1. predict.py  — a complete runnable Python script with:
                   - exact column names from training
                   - sample values from the actual dataset
                   - correct preprocessing + model loading code
                   - clear print output showing the prediction

  2. Returns a code string shown directly in the Streamlit UI
     so users can copy-paste and run immediately

Example generated predict.py:
  import joblib, pandas as pd

  # Load trained model and pipeline
  model    = joblib.load('models/lgbm_model.pkl')
  pipeline = joblib.load('models/lgbm_preprocessing.pkl')

  # Fill in your values below (based on training data)
  input_data = {
      'age':        35,        # numeric  | range: 21 - 82
      'bmi':        28.5,      # numeric  | range: 18.2 - 67.1
      'glucose':    120,       # numeric  | range: 44 - 199
      'outcome':    ...        # TARGET - do not include
  }

  # Remove target column if accidentally included
  input_df = pd.DataFrame([input_data])

  # Preprocess and predict
  X = pipeline.transform(input_df)
  prediction = model.predict(X)[0]
  print(f'Prediction: {prediction}')
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from config.settings import MODELS_DIR


class InferenceGenerator:

    def generate(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        model_name: str,
        model_path: Path,
        pipeline_path: Path,
        label_encoder=None,
    ) -> Path:
        """
        Generate a predict.py script and save it to models/.

        Parameters
        ----------
        df             : The training DataFrame (used for sample values)
        target_column  : Name of the target column
        task_type      : classification | regression
        model_name     : e.g. 'lgbm', 'xgboost'
        model_path     : Path to the saved model .pkl
        pipeline_path  : Path to the saved preprocessing .pkl
        label_encoder  : Optional LabelEncoder for target decoding

        Returns
        -------
        Path to the generated predict.py file
        """
        feature_cols = [c for c in df.columns if c != target_column]
        script       = self._build_script(
            df, feature_cols, target_column, task_type,
            model_name, model_path, pipeline_path, label_encoder
        )

        out_path = MODELS_DIR / "predict.py"
        out_path.write_text(script, encoding="utf-8")
        logger.success(f"predict.py generated -> {out_path}")
        return out_path

    def get_code_string(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        model_name: str,
        model_path: Path,
        pipeline_path: Path,
        label_encoder=None,
    ) -> str:
        """Returns the script as a string for display in Streamlit UI."""
        feature_cols = [c for c in df.columns if c != target_column]
        return self._build_script(
            df, feature_cols, target_column, task_type,
            model_name, model_path, pipeline_path, label_encoder
        )

    def _build_script(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_column: str,
        task_type: str,
        model_name: str,
        model_path: Path,
        pipeline_path: Path,
        label_encoder,
    ) -> str:
        """Build the full predict.py script content."""

        # Build column info with types and sample values
        col_info    = self._get_column_info(df, feature_cols, target_column)
        sample_dict = self._build_sample_dict(df, feature_cols)
        classes     = self._get_classes(df, target_column, task_type, label_encoder)

        # Format model path for Windows compatibility
        model_path_str    = str(model_path).replace("\\", "/")
        pipeline_path_str = str(pipeline_path).replace("\\", "/")

        # Build the input_data dict with comments
        input_lines = []
        for col in feature_cols:
            info    = col_info.get(col, {})
            dtype   = info.get("dtype", "unknown")
            comment = info.get("comment", "")
            sample  = sample_dict.get(col, "")
            # Format value
            if dtype == "numeric":
                val = repr(float(sample)) if sample != "" else "0.0"
            else:
                val = repr(str(sample)) if sample != "" else '""'
            input_lines.append(f'    "{col}": {val},  # {comment}')

        input_block = "\n".join(input_lines)

        # Build target info
        if task_type == "classification":
            target_info = f"# Predicts one of: {classes}"
        else:
            target_info = f"# Predicts a numeric value (regression)"

        script = f'''"""
predict.py
-----------
Auto-generated by Neuro AI
Model: {model_name}
Task:  {task_type}
Target column: {target_column}

HOW TO USE:
  1. Edit the values in input_data below
  2. Run: python predict.py
  3. See the prediction printed at the bottom

REQUIREMENTS:
  pip install joblib pandas scikit-learn lightgbm xgboost catboost
"""

import joblib
import pandas as pd

# ── Load trained model and preprocessing pipeline ─────────────────────────────
model    = joblib.load(r"{model_path_str}")
pipeline = joblib.load(r"{pipeline_path_str}")

# ── Fill in your input values below ───────────────────────────────────────────
# These are the EXACT columns the model was trained on.
# Replace the sample values with your own data.
{target_info}

input_data = {{
{input_block}
}}

# ── Run prediction ─────────────────────────────────────────────────────────────
input_df = pd.DataFrame([input_data])

# Apply the same preprocessing used during training
X = pipeline.transform(input_df)

# Get prediction
prediction = model.predict(X)[0]
'''

        # Add probability output for classification
        if task_type == "classification":
            script += f'''
# Get prediction probabilities (confidence)
try:
    probabilities = model.predict_proba(X)[0]
    classes = {classes}
    print("\\n=== NEURO AI PREDICTION ===")
    print(f"Prediction : {{prediction}}")
    print(f"Confidence : {{max(probabilities):.1%}}")
    print("\\nAll class probabilities:")
    for cls, prob in zip(classes, probabilities):
        bar = "#" * int(prob * 20)
        print(f"  {{cls:<20}} {{prob:.1%}}  {{bar}}")
except Exception:
    print("\\n=== NEURO AI PREDICTION ===")
    print(f"Prediction : {{prediction}}")
'''
        else:
            script += '''
print("\\n=== NEURO AI PREDICTION ===")
print(f"Predicted value : {prediction:.4f}")
'''

        script += '''
print("===========================\\n")
'''
        return script

    def _get_column_info(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_column: str,
    ) -> Dict[str, Dict]:
        """Build per-column metadata for the comments."""
        info = {}
        for col in feature_cols:
            series = df[col].dropna()
            if pd.api.types.is_numeric_dtype(series):
                mn  = round(float(series.min()), 2)
                mx  = round(float(series.max()), 2)
                avg = round(float(series.mean()), 2)
                info[col] = {
                    "dtype":   "numeric",
                    "comment": f"numeric  | range: {mn} to {mx}  | avg: {avg}",
                }
            else:
                unique_vals = series.unique().tolist()
                if len(unique_vals) <= 10:
                    vals_str = ", ".join(str(v) for v in unique_vals[:10])
                    info[col] = {
                        "dtype":   "categorical",
                        "comment": f"category | options: {vals_str}",
                    }
                else:
                    sample_vals = ", ".join(
                        str(v) for v in unique_vals[:5]
                    )
                    info[col] = {
                        "dtype":   "categorical",
                        "comment": f"category | e.g.: {sample_vals} ...",
                    }
        return info

    def _build_sample_dict(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict[str, Any]:
        """Get one representative sample row."""
        sample = df[feature_cols].dropna().iloc[0] if len(df) > 0 else {}
        result = {}
        for col in feature_cols:
            try:
                val = sample[col] if col in sample.index else ""
                # Clean quoted values
                if isinstance(val, str):
                    val = val.strip("'\"").strip()
                result[col] = val
            except Exception:
                result[col] = ""
        return result

    def _get_classes(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        label_encoder,
    ) -> List:
        """Get target class labels."""
        if task_type != "classification":
            return []
        try:
            if label_encoder is not None:
                return label_encoder.classes_.tolist()
            unique = df[target_column].dropna().unique().tolist()
            cleaned = []
            for v in unique:
                if isinstance(v, str):
                    v = v.strip("'\"").strip()
                cleaned.append(v)
            return sorted(str(v) for v in cleaned)
        except Exception:
            return []
