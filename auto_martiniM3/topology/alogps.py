"""Fragment water/octanol partition free energy (logP).

Looks fragment SMILES up in a local logP database first (cached in-process);
falls back to querying the remote ALOGPS web service (vcclab.org), also
cached per fragment, when the fragment isn't in the local file.
"""

from ..common import *

logger = logging.getLogger(__name__)

_logp_file_cache = {}
_alogps_session = requests.Session()
_alogps_cache = {}


def _read_logp_file(logp_file):
    """Parse a local fragment logP lookup file, caching the result per path
    since the file doesn't change over the course of a run."""
    if logp_file in _logp_file_cache:
        return _logp_file_cache[logp_file]
    logP_data = {}
    try:
        with open(logp_file) as f:
            for line in f:
                (key, val) = line.rstrip().split()
                logP_data[key] = float(val)
    except Exception as e:
        print(f"An error occurred while reading the logP file")
    _logp_file_cache[logp_file] = logP_data
    return logP_data


def _query_alogps(smi):
    """Query the remote ALOGPS service for a SMILES fragment, caching the
    result per fragment (many beads/attempts re-query the same fragment)."""
    if smi in _alogps_cache:
        return _alogps_cache[smi]
    req = ""
    soup = ""
    try:
        logger.debug("Calling http://vcclab.org/web/alogps/calc?SMILES=" + str(smi))
        req = _alogps_session.get(
            "http://vcclab.org/web/alogps/calc?SMILES=" + str(smi.replace("#", "%23"))
        )
    except:
        print("Error. Can't reach vcclab.org to estimate free energy.")
        exit(1)
    try:
        doc = BeautifulSoup(req.content, "lxml")
    except Exception:
        raise
    try:
        soup = doc.prettify()
    except:
        print("Error with BeautifulSoup prettify")
        exit(1)
    found_mol_1 = False
    log_p = None
    for line in soup.split("\n"):
        line = line.split()
        if "mol_1" in line:
            log_p = float(line[line.index("mol_1") + 1])
            found_mol_1 = True
            break
    _alogps_cache[smi] = (found_mol_1, log_p)
    return found_mol_1, log_p


def smi2alogps(forcepred, smi, wc_log_p, bead, converted_smi, real_smi, logp_file=None, trial=False):
    """
    Returns water/octanol partitioning free energy for a fragment: looked up
    from a local reference database when available, or predicted by ALOGPS
    (with a Wildman-Crippen fallback if forcepred is set and ALOGPS has no
    prediction for the fragment).
    """
    logger.debug("Entering smi2alogps()")

    if not logp_file:
        package_dir = os.path.dirname(os.path.dirname(__file__))
        logp_file = os.path.join(package_dir, 'logP_smi.dat')
    found_smi = False
    if bead != "MOL":
        if converted_smi:
            smi=real_smi

        # Check if logp_file is a valid file name
        if isinstance(logp_file, str) and logp_file:
            logP_data = _read_logp_file(logp_file)
        else:
            print(f"Invalid file name: {logp_file}")
            logP_data = {}

        log_p = 0.0
        for smiles, logp in logP_data.items():
            if smiles == smi:
                log_p = float(logp)
                found_smi = True
                return (log_p, "")

    if not found_smi:
        if converted_smi:
            smi=real_smi
        found_mol_1, log_p = _query_alogps(smi)
        if not found_mol_1:
            # If we're forcing a prediction, use Wildman-Crippen
            if forcepred:
                if trial:
                    wrn = (
                        "; Warning: bead ID "
                        + str(bead)
                        + " predicted from Wildman-Crippen. Fragment "
                        + str(smi)
                        + "\n"
                    )
                    sys.stderr.write(wrn)
                log_p = wc_log_p
            else:
                print("ALOGPS can't predict fragment: %s" % smi)
                exit(1)
        logger.debug("logp value: %7.4f" % log_p)
        return (convert_log_k(log_p),"; ALOGPS defined bead")


def convert_log_k(log_k):
    """Convert log_{10}K to free energy (in kJ/mol)"""
    val = 0.008314 * 300.0 * log_k / math.log10(math.exp(1))
    logger.debug("free energy %7.4f kJ/mol" % val)
    return val
