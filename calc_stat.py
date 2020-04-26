def calc_stat(train_set):

    data_mn = 0
    data_std = 0
    cnt = 0
    data_mn = train_set.data.mean(axis=0)
    data_std = train_set.data.std(axis=0)
    #for iteration, batch in enumerate(train_set,1):
    #    print('{} : {}'.format(iteration, str(batch.size())))
    #    data_mn += batch.mean(dim=0)
    #    data_std += batch.std(dim=0)
    #    cnt += 1
    #data_mn /= cnt
    #data_std /= cnt

    return data_mn, data_std
