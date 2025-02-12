# Saturation Equilibrium

Code for simulating and modeling cratered surfaces.

## Notation:

**slope**: A simulation parameter which controls the (cumulative) slope of the production function. Denoted as $b$ in the thesus.

**rmult**: A simulation parameter which controls the destructive radius of a newly-formed crater. Denoted as $R_{mult}$ in the thesis. The destructive radius of a crater is $R_{crater} \times R_{mult}$.

**erat**: A simulation parameter which controls the effectiveness of smaller craters erasing larger craters' rims. Denoted as $E_{ratio}$ in the thesis. A crater with radius $R_{new}$ only erases the rims of craters with $\E_{ratio} \geq \frac{R_{old}} {R_{new}}$.

**mrp**: A simulation parameter that controls the minimum percentage of a crater's rim that must be intact for a crater to remain in the crater_record. Denoted as $M_{r}$ in the thesis.

**rstat**: The minimum crater radius for which statistics are calculated. Denoted as $R_{stat}* in the thesis.

**nstat**: During a simulation, the total number of craters formed within the study region with $R \geq $R_{stat}$. Denoted as $N_{tot}$ in the thesis.

**nobs**: During a simulation, the number of observable craters (not destroyed) within the study region with $R \geq $R_{stat}$. Denoted as $N_{obs}$ in the thesis.

**nnd**: Nearest neighbor distance for a given crater. Denoted as $NN_d$ in the thesis.