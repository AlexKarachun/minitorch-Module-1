# MiniTorch Module 1

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module1/module1/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py tests/test_module.py tests/test_operators.py project/run_manual.py

## Test Results

```
$ python3 -m pytest tests/ -v
...
tests/test_autodiff.py::test_chain_rule1 PASSED
tests/test_autodiff.py::test_chain_rule2 PASSED
tests/test_autodiff.py::test_chain_rule3 PASSED
tests/test_autodiff.py::test_chain_rule4 PASSED
tests/test_autodiff.py::test_backprop1 PASSED
tests/test_autodiff.py::test_backprop2 PASSED
tests/test_autodiff.py::test_backprop3 PASSED
tests/test_autodiff.py::test_backprop4 PASSED
tests/test_module.py::test_stacked_demo PASSED
tests/test_module.py::test_module PASSED
tests/test_module.py::test_stacked_module PASSED
tests/test_module.py::test_module_fail_forward XFAIL
tests/test_module.py::test_module_forward PASSED
tests/test_module.py::test_parameter PASSED
tests/test_operators.py::test_same_as_python PASSED
tests/test_operators.py::test_relu PASSED
tests/test_operators.py::test_relu_back PASSED
tests/test_operators.py::test_id PASSED
tests/test_operators.py::test_lt PASSED
tests/test_operators.py::test_max PASSED
tests/test_operators.py::test_eq PASSED
tests/test_operators.py::test_sigmoid PASSED
tests/test_operators.py::test_transitive PASSED
tests/test_operators.py::test_symmetric PASSED
tests/test_operators.py::test_distribute PASSED
tests/test_operators.py::test_other PASSED
tests/test_operators.py::test_zip_with PASSED
tests/test_operators.py::test_sum_distribute PASSED
tests/test_operators.py::test_sum PASSED
tests/test_operators.py::test_prod PASSED
tests/test_operators.py::test_negList PASSED
tests/test_operators.py::test_one_args[fn0..fn13] PASSED
tests/test_operators.py::test_two_args[fn0..fn5] PASSED
tests/test_operators.py::test_backs PASSED
tests/test_scalar.py::test_central_diff PASSED
tests/test_scalar.py::test_simple PASSED
tests/test_scalar.py::test_one_args[fn0..fn13] PASSED
tests/test_scalar.py::test_two_args[fn0..fn5] PASSED
tests/test_scalar.py::test_one_derivative[fn0..fn13] PASSED
tests/test_scalar.py::test_two_derivative[fn0..fn5] PASSED

======================== 93 passed, 1 xfailed in 3.53s ========================
```

## Task 1.5 Training Log

`project/run_scalar.py`, dataset `Simple`, PTS=50, HIDDEN=2, RATE=0.5, 500 epochs:

```
Epoch  10  loss  23.59413191183285 correct 46
Epoch  20  loss  15.035833569283044 correct 48
Epoch  30  loss  13.660644463473318 correct 42
Epoch  40  loss  9.869826273376411 correct 45
Epoch  50  loss  7.655235614596173 correct 47
Epoch  60  loss  7.034172025962327 correct 47
Epoch  70  loss  6.417849114364882 correct 47
Epoch  80  loss  6.006967843080992 correct 47
Epoch  90  loss  5.609553882363652 correct 47
Epoch  100  loss  5.281299939158145 correct 47
Epoch  110  loss  4.960683612436259 correct 47
Epoch  120  loss  4.7441160826140445 correct 48
Epoch  130  loss  4.489312926762404 correct 49
Epoch  140  loss  4.2202648299532015 correct 49
Epoch  150  loss  4.025259944988225 correct 49
Epoch  160  loss  3.835864733707395 correct 49
Epoch  170  loss  3.7210756343354934 correct 49
Epoch  180  loss  3.6905894427944275 correct 49
Epoch  190  loss  3.5960785069584693 correct 49
Epoch  200  loss  3.3159315890953898 correct 49
Epoch  210  loss  3.013585960547374 correct 49
Epoch  220  loss  2.891431697429332 correct 49
Epoch  230  loss  3.0822793219524134 correct 49
Epoch  240  loss  3.9088500264717196 correct 49
Epoch  250  loss  4.137370131918049 correct 49
Epoch  260  loss  2.0866849648650625 correct 49
Epoch  270  loss  1.422228376910968 correct 50
Epoch  280  loss  1.2866304658669574 correct 50
Epoch  290  loss  1.2056700605939383 correct 50
Epoch  300  loss  1.141183610544429 correct 50
Epoch  310  loss  1.1021217493189197 correct 50
Epoch  320  loss  1.1613649506004213 correct 50
Epoch  330  loss  4.732029898503547 correct 47
Epoch  340  loss  8.5450007785299 correct 46
Epoch  350  loss  3.567870773236707 correct 49
Epoch  360  loss  1.13379561385006 correct 50
Epoch  370  loss  1.0346477864225592 correct 50
Epoch  380  loss  0.967614197415501 correct 50
Epoch  390  loss  0.9121185320569193 correct 50
Epoch  400  loss  0.8631253641329746 correct 50
Epoch  410  loss  0.8185777645697201 correct 50
Epoch  420  loss  0.777606669836293 correct 50
Epoch  430  loss  0.7397304654572704 correct 50
Epoch  440  loss  0.7046107723246963 correct 50
Epoch  450  loss  0.6719832879061414 correct 50
Epoch  460  loss  0.641615463628977 correct 50
Epoch  470  loss  0.6133088445813154 correct 50
Epoch  480  loss  0.5868840654122528 correct 50
Epoch  490  loss  0.5621812229461004 correct 50
Epoch  500  loss  0.5390568263452136 correct 50
```

`project/run_scalar.py`, dataset `Split`, PTS=50, HIDDEN=10, RATE=0.3, 500 epochs:

```
Epoch  10  loss  30.77514747369395 correct 34
Epoch  20  loss  29.85036145001201 correct 36
Epoch  30  loss  28.865527916430754 correct 39
Epoch  40  loss  27.79697577388064 correct 39
Epoch  50  loss  26.616751246634887 correct 40
Epoch  60  loss  25.329885503906326 correct 40
Epoch  70  loss  23.963480179970606 correct 39
Epoch  80  loss  28.580982823091638 correct 33
Epoch  90  loss  23.548345437712758 correct 39
Epoch  100  loss  22.29054639986223 correct 40
Epoch  110  loss  23.530345597757822 correct 40
Epoch  120  loss  20.525520931761022 correct 43
Epoch  130  loss  20.548815847001233 correct 40
Epoch  140  loss  18.479704722841642 correct 41
Epoch  150  loss  17.61243410921141 correct 42
Epoch  160  loss  15.706159634114474 correct 44
Epoch  170  loss  14.807347038987162 correct 45
Epoch  180  loss  13.20943592766167 correct 45
Epoch  190  loss  12.401873580958263 correct 45
Epoch  200  loss  10.956098827825073 correct 46
Epoch  210  loss  9.910004567380973 correct 47
Epoch  220  loss  9.588803793794485 correct 47
Epoch  230  loss  9.174249607135991 correct 47
Epoch  240  loss  7.505722153310885 correct 47
Epoch  250  loss  6.109414117545578 correct 48
Epoch  260  loss  5.133960563764727 correct 49
Epoch  270  loss  4.594174424045898 correct 49
Epoch  280  loss  4.550743591043642 correct 49
Epoch  290  loss  38.98591059252459 correct 36
Epoch  300  loss  4.568803766006661 correct 49
Epoch  310  loss  4.079115505753802 correct 50
Epoch  320  loss  3.810741111761765 correct 50
Epoch  330  loss  3.610475811213276 correct 50
Epoch  340  loss  3.4381628536608493 correct 50
Epoch  350  loss  3.2842562701203413 correct 50
Epoch  360  loss  3.145615340487312 correct 50
Epoch  370  loss  3.019402313544755 correct 50
Epoch  380  loss  2.90451578029359 correct 50
Epoch  390  loss  2.797346576784336 correct 50
Epoch  400  loss  2.70007050713218 correct 50
Epoch  410  loss  2.609030533173563 correct 50
Epoch  420  loss  2.5256269773417865 correct 50
Epoch  430  loss  2.4605514623557414 correct 50
Epoch  440  loss  2.462600039970546 correct 49
Epoch  450  loss  2.945217833868437 correct 49
Epoch  460  loss  40.66685383342699 correct 39
Epoch  470  loss  3.049979806856135 correct 50
Epoch  480  loss  2.433304822792148 correct 50
Epoch  490  loss  2.291432071291867 correct 50
Epoch  500  loss  2.1950411450966425 correct 50
```

`project/run_scalar.py`, dataset `Xor`, PTS=50, HIDDEN=10, RATE=0.1, 1000 epochs:

```
Epoch  10  loss  34.68433226949036 correct 28
Epoch  20  loss  34.450765861208986 correct 24
Epoch  30  loss  34.36832497559058 correct 25
Epoch  40  loss  34.30081212605227 correct 27
Epoch  50  loss  34.241124627758786 correct 32
Epoch  60  loss  34.17497325826598 correct 35
Epoch  70  loss  34.109207798053355 correct 37
Epoch  80  loss  34.039685685933854 correct 37
Epoch  90  loss  33.96400565244622 correct 39
Epoch  100  loss  33.88022425043145 correct 39
Epoch  150  loss  32.90002507043759 correct 40
Epoch  200  loss  30.73642827548035 correct 38
Epoch  250  loss  28.593865983790824 correct 42
Epoch  300  loss  25.910237249580224 correct 43
Epoch  350  loss  22.81704878292055 correct 43
Epoch  400  loss  19.719611587817955 correct 42
Epoch  450  loss  17.206537396372315 correct 43
Epoch  500  loss  15.334863676982 correct 43
Epoch  550  loss  13.944607352930667 correct 44
Epoch  600  loss  12.879161048702157 correct 45
Epoch  650  loss  12.16855578496342 correct 44
Epoch  700  loss  12.824622603277213 correct 43
Epoch  750  loss  11.916000589365195 correct 43
Epoch  800  loss  10.848330651026474 correct 45
Epoch  850  loss  12.114486630378305 correct 41
Epoch  900  loss  11.271970734003228 correct 44
Epoch  950  loss  9.312409956587171 correct 46
Epoch  1000  loss  11.88512206247084 correct 44
```
