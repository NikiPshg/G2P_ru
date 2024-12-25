import sys

src_path = ''

if not (src_path in sys.path ):
    sys.path.append(src_path)

from g2p_ru.g2p_ru import G2P_RU

g2p = G2P_RU()

print(g2p(" сво сво св,о 0 сво. сво-сво")) # ['s', 'v', 'o', ' ', 's', 'v', 'o', ' ', 's', 'v', ',', 'o', ' ', 's', 'v', 'o', '.', 's', 'v', 'o', '-', 's', 'v', 'o']