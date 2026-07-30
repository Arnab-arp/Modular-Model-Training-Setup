import torch

def measure_time(start, end, device=None, stepper_fn=False):
    measurement = f"{(end-start):.2f} sec"
    if stepper_fn:
        print(f'Step Run Time : {measurement}')
    else:
        print(f'''
----- Time -----
[*] Device : {device}
[*] Total Run Time : {measurement}
----------------
''')


def measure_accuracy(y_logit, y_true):
    pred_class = torch.argmax(torch.softmax(y_logit, dim=1), dim=1)
    acc = (pred_class == y_true).sum().item()/len(y_logit)
    return acc

def PerformanceGraph(results):
    pass

ints = []       
for i in range(1, 101): 
    print(i*0.75
    )