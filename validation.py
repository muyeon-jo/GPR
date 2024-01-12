from batches import get_GPR_batch,get_GPR_batch_test
import torch
import eval_metrics
from powerLaw import dist
def GeoIE_validation(model, args,num_users, positive, negative, train_matrix,val_flag,k_list,dist_mat):
    model.eval()
    recommended_list = []
    train_loss=0.0
    for user_id in range(num_users):
        user_id, target_list = get_GPR_batch_test(train_matrix,positive,negative,user_id)
        rating_ul, rating_ul_prime, e_ij_hat = model(user_id, target_list, target_list)
        # loss = model.loss_func(prediction,train_label)
        # train_loss += loss.item()
        _, indices = torch.topk(rating_ul.squeeze(), args.topk)
        recommended_list.append([target_list[i].item() for i in indices])
    
    precision, recall, hit = eval_metrics.evaluate_mp(positive,recommended_list,k_list,val_flag)
    
    return precision, recall, hit
    # return 0,[train_loss],0
