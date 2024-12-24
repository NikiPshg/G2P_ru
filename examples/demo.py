import sys

if not (r'C:/Users/RedmiBook/Apython/G2P_ru/src' in sys.path ):
    sys.path.append(r'C:/Users/RedmiBook/Apython/G2P_ru/src')

from g2p_ru.g2p_ru import G2P_RU

g2p = G2P_RU()

print(g2p('сво сво .еж : хуан хуан'))