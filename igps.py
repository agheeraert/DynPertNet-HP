fetch 1GPWC 1GPWD
remove !(polymer)
run ~/DynPertNet-HP/drawNetwork.py
set_view (\
    -0.091599651,    0.961403608,    0.259411007,\
    -0.907361209,    0.026737407,   -0.419488847,\
    -0.410239011,   -0.273805201,    0.869903326,\
     0.000195857,    0.000733256, -208.243789673,\
   -28.106067657,   12.463504791,   74.449974060,\
   165.861480713,  250.555084229,   20.000000000 )
   
drawNetwork('apo.npz', 'prfar.npz', robust_compo=True, threshold=6, label_compo=True)
