import shutil
from pathlib import Path

# RAM 
outputs_ram = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram")
outputs_ram_originale = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_originale")

shutil.copytree(outputs_ram, outputs_ram_originale)

# RAM FILTRATO
outputs_ram_clean = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_clean")
outputs_ram_clean_originale = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_clean_originale")

shutil.copytree(outputs_ram_clean, outputs_ram_clean_originale)

# RAM FILTRATO ITA
outputs_ram_clean_ita = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_clean_ita")
outputs_ram_clean_ita_originale = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_clean_ita_originale")

shutil.copytree(outputs_ram_clean_ita, outputs_ram_clean_ita_originale)

# SAM3
outputs_sam3_tutti = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_sam3_tutti")
outputs_sam3_tutti_originale = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_sam3_tutti_originale")

shutil.copytree(ooutputs_sam3_tutti, outputs_sam3_tutti_originale)