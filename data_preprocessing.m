% Inertial data

% training data
start_of_actions = 24
end_of_actions = 26
end_of_subjects = 7
num_of_timesteps = 4
sensor_data =  'inertial'
X_iner = {}
arr_combined_train = []

for i = start_of_actions : end_of_actions
    for j = 1 : end_of_subjects
        for k = 1 : num_of_timesteps
            load('a'+string(i)+'_s'+string(j)+'_t'+string(k)+'_'+sensor_data+'.mat')
            val = d_iner(:,:)
            valMA = movmean(val,4)
            val(:,7) = i
            X_iner(end+1) = {val}
            arr_combined_train = [arr_combined_train; val]
        end
    end
end

X_iner = transpose(X_iner)

% test data
start_of_actions = 24
end_of_actions = 26
start_of_subjects = 8
end_of_subjects = 8
num_of_timesteps = 4
sensor_data = 'inertial'
X_test = {}
arr_combined_test = []
arr_combined_test_response = []

for i = start_of_actions : end_of_actions
    for j = start_of_subjects : end_of_subjects
        for k = 1 : num_of_timesteps
            load('a'+string(i)+'_s'+string(j)+'_t'+string(k)+'_'+sensor_data+'.mat')
            val = d_iner(:,:)
            X_test(end+1) = {val}
            val_with_response = val
            val_with_response(:,7) = i
            arr_combined_test_response = [arr_combined_test_response; val_with_response]
            arr_combined_test = [arr_combined_test; val]
        end
    end
end

X_test = transpose(X_test)
