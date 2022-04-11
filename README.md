# EECS545_Final_Proj

### TO DO
- move Synapse under datasets directory

### Command to Train
```
python train_isic.py
```

### Command to Test
```
python test.py --test_model=Transfuse_epoch_19.pth --is_savenii
```

### Problems So Far

##### Training:
```
iteration 302 : loss : 0.322042, loss_ce: 0.121768
Traceback (most recent call last):
  File "/home/marco/Documents/TransFuse/train_isic.py", line 148, in <module>
    best_loss = train(train_loader, model, optimizer, epoch, best_loss, opt, epoch)
  File "/home/marco/Documents/TransFuse/train_isic.py", line 53, in train
    optimizer.step()
  File "/home/marco/anaconda3/lib/python3.9/site-packages/torch/optim/optimizer.py", line 89, in wrapper
    return func(*args, **kwargs)
  File "/home/marco/anaconda3/lib/python3.9/site-packages/torch/autograd/grad_mode.py", line 27, in decorate_context
    return func(*args, **kwargs)
  File "/home/marco/anaconda3/lib/python3.9/site-packages/torch/optim/adam.py", line 108, in step
    F.adam(params_with_grad,
  File "/home/marco/anaconda3/lib/python3.9/site-packages/torch/optim/_functional.py", line 96, in adam
    param.addcdiv_(exp_avg, denom, value=-step_size)
RuntimeError: value cannot be converted to type float without overflow: (1.04646e-05,-3.40017e-06)
```

**Possible Solution**: adjust learning rate

##### Testing:
```
idx 0 case case0008 mean_dice 0.599285 mean_hd95 14.418722
1it [02:22, 142.47s/it]idx 1 case case0022 mean_dice 0.753854 mean_hd95 7.699350
2it [03:48, 109.18s/it]idx 2 case case0038 mean_dice 0.683122 mean_hd95 8.860537
3it [05:29, 105.48s/it]Killed
```


**Possible Cause of Problem**: DRAM too small
