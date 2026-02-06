from Bio import SeqIO
from loguru import logger
import subprocess as sp
from pathlib import Path


def generate_foldseek_db(in_fasta, in_3di, outdir, db_name):

    outdir = Path(f"{outdir}/tmp")


    try:
        process = sp.Popen(["foldseek", "version"], stdout=sp.PIPE, stderr=sp.STDOUT)
    except:
        logger.error("Foldseek not found. Please reinstall phold.")
    
    foldseek_out, _ = process.communicate()
    foldseek_out = foldseek_out.decode()

    foldseek_version = foldseek_out.strip()

    logger.info(
        f"Foldseek version found is v{foldseek_version}"
    )


    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # read amino-acid sequences (preserve order)
    sequences_aa = {}
    for record in SeqIO.parse(in_fasta, "fasta"):
        sequences_aa[record.id] = str(record.seq)

    # read 3Di strings
    sequences_3di = {}
    for record in SeqIO.parse(in_3di, "fasta"):
        if record.id not in sequences_aa:
            logger.warning(
                f"Ignoring 3Di entry {record.id}, since it is not in the amino-acid FASTA file"
            )
            continue
        sequences_3di[record.id] = str(record.seq).upper()

    # sanity check
    missing = set(sequences_aa) - set(sequences_3di)
    if missing:
        raise ValueError(
            f"The following AA entries have no corresponding 3Di string: "
            f"{', '.join(list(missing)[:5])}..."
        )

    aa_tsv = tmp_dir / "aa.tsv"
    di_tsv = tmp_dir / "3di.tsv"
    header_tsv = tmp_dir / "header.tsv"
    lookup_path = Path(f"{db_name}.lookup")

    # generate TSV files
    with (
        open(aa_tsv, "w") as f_aa,
        open(di_tsv, "w") as f_3di,
        open(header_tsv, "w") as f_header,
        open(lookup_path, "w") as f_lookup,
    ):
        for i, seq_id in enumerate(sequences_aa, start=1):
            f_aa.write(f"{i}\t{sequences_aa[seq_id]}\n")
            f_3di.write(f"{i}\t{sequences_3di[seq_id]}\n")
            f_header.write(f"{i}\t{seq_id}\n")
            f_lookup.write(f"{i}\t{seq_id}\t0\n")

    # create Foldseek databases
    sp.run(
        ["foldseek", "tsv2db", str(aa_tsv), db_name, "--output-dbtype", "0"],
        check=True,
    )
    sp.run(
        ["foldseek", "tsv2db", str(di_tsv), f"{db_name}_ss", "--output-dbtype", "0"],
        check=True,
    )
    sp.run(
        ["foldseek", "tsv2db", str(header_tsv), f"{db_name}_h", "--output-dbtype", "12"],
        check=True,
    )

    # clean up temp files
    for p in (aa_tsv, di_tsv, header_tsv):
        p.unlink(missing_ok=True)
