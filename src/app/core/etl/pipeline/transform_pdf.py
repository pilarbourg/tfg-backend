import re
import os
import pandas as pd
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

PIPELINE_OPTIONS = PdfPipelineOptions()
PIPELINE_OPTIONS.do_ocr = True
PIPELINE_OPTIONS.do_table_structure = True

CONVERTER = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=PIPELINE_OPTIONS)
    }
)

def process_pdf(pmcid : str) -> str | None:
    """
    Processes the PubMed article in PDF format to ensure tabular data and chemical nomenclature are maintained, especially in the case of complex metabolic data.

    Parameters
    ----------
    pmcid : str
        PubMed Central identifier for the paper.

    Returns
    -------
    str or None
        Path to the processed markdown file on success, or None if processing fails.
    """

    pdf_path = f"downloads/PMC{pmcid}.pdf"
    output_path = f"results/PMC{pmcid}.md"

    os.makedirs("results", exist_ok=True)

    if not os.path.exists(pdf_path):
        logging.error("Path to file does not exist.")
        return None

    try:
        result = CONVERTER.convert(pdf_path)
        doc = result.document
        markdown = doc.export_to_markdown()

        cleaned = re.sub(r'<!-- image -->', '', markdown)
        cleaned = re.split(r'\n#+\s*\d*\.?\s*Introduction', cleaned, flags=re.IGNORECASE, maxsplit=1)
        cleaned = '## Introduction\n' + cleaned[1] if len(cleaned) > 1 else cleaned[0]
        cleaned = re.split(r'\n#+\s*(References|Bibliography|Works Cited)', cleaned, flags=re.IGNORECASE)[0]
        cleaned = re.sub(r'Figure\s+\d+\..*?(?=\n\n|\Z)', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r',\s*x FOR PEER REVIEW', '', cleaned)
        cleaned = re.sub(r'\b\d+\s+of\s+\d+\b', '', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        paragraphs = cleaned.split('\n\n')
        seen = []
        for p in paragraphs:
            stripped = p.strip()
            if stripped and stripped not in seen:
                seen.append(stripped)
        deduped = '\n\n'.join(seen)

        lines = deduped.split('\n')
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('#') or
                    stripped.startswith('|') or
                    stripped == '' or
                    len(stripped) > 30 or
                    stripped.endswith('.')):
                clean_lines.append(line)
        deduped = '\n'.join(clean_lines)
        deduped = re.sub(r'\n{3,}', '\n\n', deduped)
        deduped = re.sub(r'\n\s*[-\d\.,·:]+\s*\n', '\n', deduped)
        deduped = re.sub(r'\n[A-Z]\s*\n', '\n', deduped)

        tables_md = []
        for i, table in enumerate(doc.tables):
            try:
                df = table.export_to_dataframe(doc=doc)
                df.columns = range(len(df.columns))
                expanded_rows = []
                for _, row in df.iterrows():
                    first_col = str(row.iloc[0])
                    if '\n' in first_col:
                        parts = first_col.split('\n')
                        for part in parts:
                            new_row = row.copy()
                            new_row.iloc[0] = part.strip()
                            expanded_rows.append(new_row)
                    else:
                        expanded_rows.append(row)
                df = pd.DataFrame(expanded_rows).reset_index(drop=True)
                tables_md.append(f"## Table {i+1}\n\n{df.to_markdown(index=False)}")
            except Exception as e:
                logging.error(f"Error processing table {i+1}: {e}")
                continue

        final_output = deduped
        if tables_md:
            final_output += '\n\n' + '\n\n'.join(tables_md)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_output)
    
        os.remove(pdf_path)
        logging.info(f"Deleted temporary PDF: {pdf_path}")
        return output_path

    except Exception as e:
        logging.error(f"Failed PMC{pmcid}: {e}")
