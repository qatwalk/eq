
**QATWALK** welcomes all models - the jumps, the roughs, and fat tails...

![cats](images/three_cats.png)

## Premise
In *“A Perfect Calibration ! Now What ?,”* 2003, Schoutens et al compared 7 volatility models to show that -
though the models were calibrated to the same vanilla prices, they produced different results for exotics such as cliquets, barriers, etc.

Since then, many models have appeared in the landscape, such as the Bergomi, Rough Bergomi and Rough Heston models. There have been studies on the processes, calibration methods, and their implied vol surfaces, but no comprehensive study on their impact on different exotic contracts.

## Models
This repositary aims to be an open, collaborative, and growing study. It contains a number of models, with parameters calibrated to a common volatility surface. See the [comparison of vanilla surfaces](01a_VanillaSurface.ipynb) as a validation of the calibration.

![vanilla](images/vanilla.png)

## Contracts
It contains a series of notebooks, for different contracts such as [cliquets](02_Cliquet.ipynb), [collars](03a_Collars.ipynb), [forward collars](03b_FwdCollars.ipynb), and [lookback options](04b_Lookbacks_Model_Prices.ipynb), to compare prices from each model and gain insights of their specific dynamics.


<table>
<tr>
<td> <img src="images/collar.png" alt="collar"/> </td>
<td> <img src="images/fwd collar.png" alt="forward collar"/> </td>
</tr>
<tr>
<td> <img src="images/lookback_put.png" alt="lookback"/> </td>
<td> <img src="images/cliquet.png" alt="cliquet"/> </td>
</tr>
</table>


We have started with a few, and we will let the models, and contracts grow over time with more contributors.

## How to Contribute
You can contribute by

- writing new models (e.g. see [Bergomi](src/model/rbergomi.py))
- writing new contracts, (e.g. see [forward collars](03b_FwdCollars.ipynbipynb))
- calibration methods

Start by looking at the [discussions](https://github.com/qatwalk/eq/discussions) and [issues](https://github.com/qatwalk/eq/issues).