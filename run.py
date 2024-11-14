from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "p1"
    tasks = {
        "algorithms" : ["bsnet","bsnet2","bsnet2sc","bsnet2entropy","bsnet2both"],
        "datasets": ["indian_pines"],
        "target_sizes" : list(range(30,1,-1))
    }
    ev = TaskRunner(tasks,tag, verbose=True)
    summary, details = ev.evaluate()
