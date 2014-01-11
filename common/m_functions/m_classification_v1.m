%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% m_classification
% classification using the video feature
% output:
% kernel_mat_train_s%02d_[feature_type].mat
% result.mat
% -----------------------------------------------------------------------
% version information:
% v1: add the method of computing kernel mat, load all the video descripors, and compute the kernel mat, but need more memory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function return_state = m_classification_v1(global_config)
return_state = 1;

% load dataset info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx',
% 'train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);

classification_config = global_config.classification;
splits = classification_config.splits;
descriptor_type = classification_config.descriptor_type;

descriptor_weight = classification_config.descriptor_weight;

% get which classes to classification
switch class(global_config.classification.class_idx)
    case 'double'
        do_class_idx = global_config.classification.class_idx;
        
    case 'char'
        do_class_idx = [1:class_num];
        
    otherwise
        fprintf('WRONG!!! wrong type of global_config.extract_features.class_idx!\n');
        return_state = 0;
        return;
end




for i_splits = 1:length(splits)
    cur_splits = splits(i_splits);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load or compute the kernel mat
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    total_train_kernel_mat = zeros(train_num_per_class*length(do_class_idx),train_num_per_class*length(do_class_idx));
    total_test_kernel_mat = zeros(test_num_per_class*length(do_class_idx),train_num_per_class*length(do_class_idx));
    
    kernelmat_num = 0;
    if ~classification_config.kernelmat.recompute
        % do not recompute the kernel mat, only reload
        
        for i_desc = 1:length(descriptor_type)
            cur_descriptor = descriptor_type{i_desc};
            cur_weight = descriptor_weight(i_desc);
            
            switch cur_descriptor
                case 'low_level'
                    feature_types = classification_config.feature_types_low;
                case 'mid_level'
                    feature_types = classification_config.feature_types_mid;
                otherwise
                    fprintf('wrong descriptor type! \n');
                    return;
            end
            
            
            for i_feature = 1:length(feature_types)
                cur_feature = feature_types{i_feature};
                
                fprintf('loading kernal mat splits:%2d desc:%s feat:%s\n',cur_splits,cur_descriptor,cur_feature);
                
                load_name = fullfile(classification_config.path,sprintf('kernel_mat_train_s%02d_%s_%s.mat', cur_splits,cur_descriptor,cur_feature));
                if ~exist(load_name,'file')
                    fprintf('WRONG!!! no descriptor exist! %s\n',load_name);
                    return_state = 0;
                    return;
                end
                load(load_name,'train_kernel_mat');
                
                load_name = fullfile(classification_config.path,sprintf('kernel_mat_test_s%02d_%s_%s.mat', cur_splits,cur_descriptor,cur_feature));
                if ~exist(load_name,'file')
                    fprintf('WRONG!!! no descriptor exist! %s\n',load_name);
                    return_state = 0;
                    return;
                end
                load(load_name,'test_kernel_mat');
                kernelmat_num = kernelmat_num +1;
                total_train_kernel_mat = total_train_kernel_mat + train_kernel_mat*cur_weight;
                total_test_kernel_mat = total_test_kernel_mat + test_kernel_mat*cur_weight;
                
            end % i_feature = 1:length(feature_types)
        end % i_desc = 1:length(descriptor_type)
        
    else % ~classification_config.kernelmat.recompute
        % compute the kernel mat
        
        for i_desc = 1:length(descriptor_type)
            cur_descriptor = descriptor_type{i_desc};
            cur_weight = descriptor_weight(i_desc);
            switch cur_descriptor
                case 'low_level'
                    desc_path = global_config.extract_low_level_descriptors.path;
                case 'mid_level'
                    desc_path = global_config.extract_mid_level_descriptors.path;
                otherwise
                    fprintf('WRONG!!! unknown video feature type! %s\n',classification_config.feature);
                    return_state = 0;
                    return;
            end
            
            switch classification_config.kernelmat.method
                case 'low_mem'
                    % generate the compute params
                    % train kernel mat params
                    i_cell = 0;
                    
                    %every low is a group param to compute the kenelmat at location
                    %(params{i,1},params{i,2}), and the related class id is ( params{i,3},
                    %params{i,4} ), and the related videos index vector id is ( params{i,5},params{i,6} )
                    train_kernel_params = {};
                    for i = 1:length(do_class_idx)
                        
                        for j = i:length(do_class_idx)
                            i_cell = i_cell +1;
                            train_kernel_params{i_cell,1} = i;
                            train_kernel_params{i_cell,2} = j;
                            
                            train_kernel_params{i_cell,3} = do_class_idx(i);
                            train_kernel_params{i_cell,4} = do_class_idx(j);
                            
                            train_kernel_params{i_cell,5} = train_set_idx{cur_splits}{do_class_idx(i)};
                            train_kernel_params{i_cell,6} = train_set_idx{cur_splits}{do_class_idx(j)};
                            
                        end
                    end
                    
                    % test kernel mat params
                    i_cell = 0;
                    test_kernel_params = {};
                    for i = 1:length(do_class_idx)
                        
                        for j = 1:length(do_class_idx)
                            i_cell = i_cell +1;
                            test_kernel_params{i_cell,1} = i;
                            test_kernel_params{i_cell,2} = j;
                            
                            test_kernel_params{i_cell,3} = do_class_idx(i);
                            test_kernel_params{i_cell,4} = do_class_idx(j);
                            
                            test_kernel_params{i_cell,5} = test_set_idx{cur_splits}{do_class_idx(i)};
                            test_kernel_params{i_cell,6} = train_set_idx{cur_splits}{do_class_idx(j)};
                            
                        end
                    end
                    
                    
                    
                    for i_feature = 1:length(feature_types)
                        cur_feature = feature_types{i_feature};
                        
                        if global_config.num_core > 1
                            parfor i_cell = 1:size(train_kernel_params,1)
                                
                                train_kernel_rec{i_cell} = ...
                                    compute_linear_kernel_dis(cur_splits,cur_feature,train_kernel_params{i_cell,3},train_kernel_params{i_cell,4},...
                                    train_kernel_params{i_cell,5},train_kernel_params{i_cell,6},...
                                    cur_descriptor,desc_path,classification_config.rho);
                            end
                            
                            parfor i_cell = 1:size(test_kernel_params,1)
                                test_kernel_rec{i_cell} = ...
                                    compute_linear_kernel_dis(cur_splits,cur_feature,test_kernel_params{i_cell,3},test_kernel_params{i_cell,4},...
                                    test_kernel_params{i_cell,5},test_kernel_params{i_cell,6},...
                                    cur_descriptor,desc_path,classification_config.rho);
                            end
                        else
                            for i_cell = 1:size(train_kernel_params,1)
                                
                                train_kernel_rec{train_kernel_params{i_cell,1},train_kernel_params{i_cell,2}} = ...
                                    compute_linear_kernel_dis(cur_splits,cur_feature,train_kernel_params{i_cell,3},train_kernel_params{i_cell,4},...
                                    train_kernel_params{i_cell,5},train_kernel_params{i_cell,6},...
                                    cur_descriptor,desc_path,classification_config.rho);
                                
                            end
                            
                            for i_cell = 1:size(test_kernel_params,1)
                                
                                test_kernel_rec{i_cell} = ...
                                    compute_linear_kernel_dis(cur_splits,cur_feature,test_kernel_params{i_cell,3},test_kernel_params{i_cell,4},...
                                    test_kernel_params{i_cell,5},test_kernel_params{i_cell,6},...
                                    cur_descriptor,desc_path,classification_config.rho);
                            end
                        end
                        
                        train_kernel_cell = cell(length(do_class_idx));
                        test_kernel_cell = cell(length(do_class_idx));
                        
                        for i_cell = 1:size(train_kernel_params,1)
                            train_kernel_cell{ train_kernel_params{i_cell,1},train_kernel_params{i_cell,2} } = train_kernel_rec{i_cell};
                            train_kernel_cell{ train_kernel_params{i_cell,2},train_kernel_params{i_cell,1} } = train_kernel_rec{i_cell}';
                        end
                        
                        for i_cell = 1:size(test_kernel_params,1)
                            test_kernel_cell{ test_kernel_params{i_cell,1},test_kernel_params{i_cell,2} } = test_kernel_rec{i_cell};
                        end
                        
                        train_kernel_mat = cell2mat(train_kernel_cell);
                        test_kernel_mat = cell2mat(test_kernel_cell);
                        
                        kernelmat_num = kernelmat_num +1;
                        total_train_kernel_mat = total_train_kernel_mat + train_kernel_mat;
                        total_test_kernel_mat = total_test_kernel_mat + test_kernel_mat;
                        
                        save_name = fullfile(classification_config.path,sprintf('kernel_mat_train_s%02d_%s_%s.mat', cur_splits,cur_descriptor,cur_feature));
                        save(save_name,'train_kernel_mat');
                        
                        save_name = fullfile(classification_config.path,sprintf('kernel_mat_test_s%02d_%s_%s.mat', cur_splits,cur_descriptor,cur_feature));
                        save(save_name,'test_kernel_mat');
                        
                    end % i_feature = 1:length(feature_types)
                    
                case 'high_mem'
                    
                    
                    switch cur_descriptor
                        case 'low_level'
                            feature_types = classification_config.feature_types_low;
                        case 'mid_level'
                            feature_types = classification_config.feature_types_mid;
                        otherwise
                            fprintf('wrong descriptor type! \n');
                            return;
                    end
                    
                    for i_feature = 1:length(feature_types)
                        cur_feature = feature_types{i_feature};
                                                
                        
                        load_name = fullfile(desc_path,sprintf('s%02d_c%03d_v%03d_%s.mat', cur_splits,1,1,cur_feature) );
                        load(load_name);
                        switch cur_descriptor
                            case 'low_level'
                                descriptor = ll_descriptor;
                                
                            case 'mid_level'
                                if (classification_config.rho > 0)
                                    ml_descriptor = 1./(1+exp(-classification_config.rho*ml_descriptor));
                                end
                                descriptor = ml_descriptor;
                                
                            otherwise
                                fprintf('WRONG!!! unknown video feature type! %s\n',classification_config.feature);
                                return;
                                
                        end
                        
                        train_data = zeros(train_num_per_class*length(do_class_idx),length(descriptor));
                        test_data = zeros(test_num_per_class*length(do_class_idx),length(descriptor));
                        
                        i_row = 0;
                        for i_class = 1:length(do_class_idx)
                            cur_class = do_class_idx(i_class);
                            fprintf('train loading split:%2d desc:%s feat:%s class:%3d\n',cur_splits,cur_feature,cur_feature,cur_class);
                            
                            vids_idx = train_set_idx{cur_splits}{cur_class};
                                                       
                            for i_vid = 1:length(vids_idx)
                                cur_vid = vids_idx(i_vid);
                                
                                
                                
                                load_name = fullfile(desc_path,sprintf('s%02d_c%03d_v%03d_%s.mat', cur_splits,cur_class,cur_vid,cur_feature) );
                                load(load_name);
                                
                                
                                switch cur_descriptor
                                    case 'low_level'
                                        descriptor = ll_descriptor;
                                        
                                    case 'mid_level'
                                        if (classification_config.rho > 0)
                                            ml_descriptor = 1./(1+exp(-classification_config.rho*ml_descriptor));
                                        end
                                        descriptor = ml_descriptor;
                                        
                                    otherwise
                                        fprintf('WRONG!!! unknown video feature type! %s\n',classification_config.feature);
                                        return;
                                        
                                end
                                
                                % l2 normalize
                                descriptor = descriptor/norm(descriptor,2);
                                
                                i_row = i_row +1;
                                train_data(i_row,:) = descriptor;
                                
                            end
                            
                        end
                        
                        i_row = 0;
                        for i_class = 1:length(do_class_idx)
                            cur_class = do_class_idx(i_class);
                            
                            fprintf('test loading split:%2d desc:%s feat:%s class:%3d\n',cur_splits,cur_feature,cur_feature,cur_class);
                            
                            vids_idx = test_set_idx{cur_splits}{cur_class};
                            
                            for i_vid = 1:length(vids_idx)
                                cur_vid = vids_idx(i_vid);
                                
                                
                                
                                load_name = fullfile(desc_path,sprintf('s%02d_c%03d_v%03d_%s.mat', cur_splits,cur_class,cur_vid,cur_feature) );
                                load(load_name);
                                
                                switch cur_descriptor
                                    case 'low_level'
                                        descriptor = ll_descriptor;
                                        
                                    case 'mid_level'
                                        if (classification_config.rho > 0)
                                            ml_descriptor = 1./(1+exp(-classification_config.rho*ml_descriptor));
                                        end
                                        descriptor = ml_descriptor;
                                        
                                        
                                    otherwise
                                        fprintf('WRONG!!! unknown video feature type! %s\n',classification_config.feature);
                                        return;                                       
                                end
                                
                                % l2 normalize
                                descriptor = descriptor/norm(descriptor);
                                
                                i_row = i_row +1;
                                test_data(i_row,:) = descriptor;
                                
                            end
                            
                        end
                        
                        train_kernel_mat = train_data*train_data';
                        test_kernel_mat = test_data*train_data';
                        
                        kernelmat_num = kernelmat_num +1;
                        total_train_kernel_mat = total_train_kernel_mat + train_kernel_mat*cur_weight;
                        total_test_kernel_mat = total_test_kernel_mat + test_kernel_mat*cur_weight;
                        
                        save_name = fullfile(classification_config.path,sprintf('kernel_mat_train_s%02d_%s_%s.mat', cur_splits,cur_descriptor,cur_feature));
                        save(save_name,'train_kernel_mat');
                        
                        save_name = fullfile(classification_config.path,sprintf('kernel_mat_test_s%02d_%s_%s.mat', cur_splits,cur_descriptor,cur_feature));
                        save(save_name,'test_kernel_mat');
                        
                    end % i_feature = 1:length(feature_types)
                    
                    
                    
                otherwise
                    fprintf('WRONG!!! no classification_config.kernelmat.method! %s\n',classification_config.kernelmat.method);
                    return_state = 0;
                    return;
                    
            end %  switch classification_config.kernelmat.method
            
            
            
        end % i_desc = 1:length(descriptor_type)
        
    end % if ~classification_config.recompute_kernelmat
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % classification
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    total_train_kernel_mat = total_train_kernel_mat/kernelmat_num;
    total_test_kernel_mat = total_test_kernel_mat/kernelmat_num;
    
    train_acc = [];
    test_acc = [];
    
    models = {};
    train_pre_vals = [];
    test_pre_vals = [];
    
    train_label = [];
    test_label = [];
    n_class = length(do_class_idx);
    for i = 1:n_class
        train_label = [train_label; i*ones( length( train_set_idx{cur_splits}{do_class_idx(i)}) ,1 )];
        test_label = [test_label; i*ones( length( test_set_idx{cur_splits}{do_class_idx(i)}) ,1 )];
    end
    
    
    svm_kernel_train = [(1:size(total_train_kernel_mat,1))',total_train_kernel_mat];
    svm_kernel_test = [(1:size(total_test_kernel_mat,1))',total_test_kernel_mat];
    
    
    for i_calss = 1:n_class
        %generate label
        cur_train_label = (train_label == i_calss) - (train_label~=i_calss);
        cur_test_label = (test_label == i_calss) - (test_label~=i_calss);
        
        libsvm_params = sprintf('-t 4 -c %d -q',classification_config.libsvm_params.C);
        
        cur_model = svmtrain(cur_train_label, svm_kernel_train, libsvm_params);
        models{i_calss} = cur_model;
        
        [~, cur_train_accuracy, cur_train_preval] = svmpredict(cur_train_label,svm_kernel_train,cur_model);
        train_pre_vals = [train_pre_vals,cur_train_preval*cur_model.Label(1)];
        train_acc = [train_acc;cur_train_accuracy(1)];
        
        [~, cur_test_accuracy, cur_test_preval] = svmpredict(cur_test_label,svm_kernel_test,cur_model);
        test_pre_vals = [test_pre_vals,cur_test_preval*cur_model.Label(1)];
        test_acc = [test_acc;cur_test_accuracy(1)];
        
        %     curtrain_label = (train_label == iClass) - (train_label~=iClass);
        %     curtest_label = (test_label == iClass) - (test_label~=iClass);
        %     curModel = svmtrain(curtrain_label, trainDataAll, '-t 2 -c 1 -g 10');
        %     [curTrainPreLabel, curTrainAccuracy, curTrainpreVal] = svmpredict(curtrain_label,trainDataAll,curModel);
        %     trainAcc = [trainAcc;curTrainAccuracy(1)];
        %     [curTestPreLabel, curTestAccuracy, curTestpreVal] = svmpredict(curtest_label,testDataAll,curModel);
        %     testAcc = [testAcc;curTestAccuracy(1)];
    end
    
    %sort as the predict score
    [~,sortIdx] = sort(train_pre_vals,2,'descend');
    train_pre_idx = sortIdx(:,1);
    train_total_acc = sum(train_pre_idx == train_label)/length(train_label);
    train_each_acc = sum(reshape(train_pre_idx,[],n_class) == reshape(train_label,[],n_class))/train_num_per_class;
    
    [~,sortIdx] = sort(test_pre_vals,2,'descend');
    test_pre_idx = sortIdx(:,1);
    test_total_acc = sum(test_pre_idx == test_label)/length(test_label);
    test_each_acc = sum(reshape(test_pre_idx,[],n_class) == reshape(test_label,[],n_class))/test_num_per_class;
    
    
    train_total_acc
    train_each_acc
    test_total_acc
    test_each_acc
    
    % compute confusion matrix
    confusion_matrixt = compute_confusion_matrix(test_pre_idx,test_num_per_class*ones(n_class,1),class_names);
    
    save_name = fullfile(classification_config.path,sprintf('s%02d_results.mat',cur_splits) );
    save(save_name,'train_total_acc','train_each_acc','test_total_acc','test_each_acc','confusion_matrixt','class_names');
    
end % i_splits = 1:length(splits)


end

%%

function  kernel_mat = compute_linear_kernel_dis(cur_splits,cur_feature,i_class,j_class,i_class_vids_idx,j_class_vids_idx,cur_decriptor,vid_descriptor_path,rho)


fprintf('computing kernel mat of: split:%2d feature:%s i_class:%3d j_class:%3d\n',cur_splits,cur_feature,i_class,j_class);

% load all the data needed
i_class_feat_data = [];
for i = 1:length(i_class_vids_idx)
    load_name = fullfile(vid_descriptor_path,sprintf('s%02d_c%03d_v%03d_%s.mat', cur_splits,i_class,i_class_vids_idx(i),cur_feature) );
    load(load_name);
    
    switch cur_decriptor
        case 'low_level'
            descriptor = ll_descriptor;
            
        case 'mid_level'
            if (classification_config.rho > 0)
                ml_descriptor = 1./(1+exp(-classification_config.rho*ml_descriptor));
            end
            descriptor = ml_descriptor;
            
        otherwise
            fprintf('WRONG!!! unknown video feature type! %s\n',cur_decriptor);
            return;
            
    end
    
    % l2 normalize
    descriptor = descriptor/norm(descriptor);
    i_class_feat_data = [i_class_feat_data;descriptor];
end

j_class_feat_data = [];

for i = 1:length(j_class_vids_idx)
    load_name = fullfile(vid_descriptor_path,sprintf('s%02d_c%03d_v%03d_%s.mat', cur_splits,j_class,j_class_vids_idx(i),cur_feature) );
    load(load_name);
    
    switch cur_decriptor
        case 'low_level'
            descriptor = ll_descriptor;
            
        case 'mid_level'
            
            if (classification_config.rho > 0)
                ml_descriptor = 1./(1+exp(-classification_config.rho*ml_descriptor));
            end
            descriptor = ml_descriptor;
            
        otherwise
            fprintf('WRONG!!! unknown video feature type! %s\n',cur_decriptor);
            return;
            
    end
    
    % l2 normalize
    descriptor = descriptor/norm(descriptor);
    j_class_feat_data = [j_class_feat_data;descriptor];
end



kernel_mat = i_class_feat_data*j_class_feat_data';

end