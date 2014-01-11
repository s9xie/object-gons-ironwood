%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% m_classification
% classification using the video feature
% output:
% kernel_mat_train_s%02d_[feature_type].mat
% result.mat
% -----------------------------------------------------------------------
% version information:
% v1: add the method of computing kernel mat, load all the video descripors, and compute the kernel mat, but need more memory
% v2: support differnt rho
% v3: support sparse LDA and change to support other dataset
% v4: liblinear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function return_state = m_classification_v3(global_config)
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

if global_config.classification.do_sparseLDA
    do_sparseLDA_v0(global_config);
end




for i_splits = 1:length(splits)
    cur_splits = splits(i_splits);
    
    
    % load sparse 
    if classification_config.using_sparseLDA
        load_name = fullfile(classification_config.path,sprintf('s%02d_sparse_weight.mat',cur_splits) );
        load(load_name,'BetaTab','ThetaTab','AlphaTab','lambda','sparse_weight','sparse_base');
%         feat_num = 15;
%         sparse_base = cell(feat_num,1);
%         feat_dim = size(BetaTab{end},1)/feat_num;
%         for i_feat = 1:feat_num
%             sparse_base{i_feat} = BetaTab{end}(feat_dim*(i_feat-1)+1:feat_dim*i_feat,:);
%         end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load or compute the kernel mat
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    train_num_tot = 0;
    test_num_tot = 0;
    train_num_each_class = [];
    test_num_each_class = [];
    for i_class = 1:length(do_class_idx)
        cur_class = do_class_idx(i_class);
        
        train_num_each_class = [train_num_each_class,length( train_set_idx{cur_splits}{cur_class} )];
        test_num_each_class = [test_num_each_class,length( test_set_idx{cur_splits}{cur_class} )];
    end
    
    train_num_tot = sum(train_num_each_class);
    test_num_tot = sum(test_num_each_class);
    
    total_train_kernel_mat = zeros( train_num_tot,train_num_tot );
    total_test_kernel_mat = zeros( test_num_tot,train_num_tot );
    
    kernelmat_num = 0;
    mid_level_feat_count = 0;
    tot_weight = 0;
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
                tot_weight = tot_weight+cur_weight;
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
                if length(classification_config.rho) > 1
                    cur_rho = classification_config.rho(mid_level_feat_count +1);
                else
                    cur_rho = classification_config.rho(1);
                end
                
                load_name = fullfile(desc_path,sprintf('s%02d_c%03d_v%03d_%s.mat', cur_splits,1,1,cur_feature) );
                load(load_name);
                switch cur_descriptor
                    case 'low_level'
                        descriptor = ll_descriptor;
                        
                    case 'mid_level'
                        if (classification_config.rho >= 0)
                            ml_descriptor = 1./(1+exp(-cur_rho*ml_descriptor));
                        end
                        descriptor = ml_descriptor;
                        mid_level_feat_count = mid_level_feat_count+1;
                    otherwise
                        fprintf('WRONG!!! unknown video feature type! %s\n',classification_config.feature);
                        return;
                        
                end
                
                
                
                
                train_data = zeros(train_num_tot,length(descriptor));
                test_data = zeros(test_num_tot,length(descriptor));
                
                
                
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
                                if (classification_config.rho >= 0)
                                    ml_descriptor = 1./(1+exp(-cur_rho*ml_descriptor));
                                end
                                descriptor = ml_descriptor;
                                
                            otherwise
                                fprintf('WRONG!!! unknown video feature type! %s\n',classification_config.feature);
                                return;
                                
                        end
                        
                        % l2 normalize
%                         descriptor = descriptor/norm(descriptor,2);
                        
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
                                if (classification_config.rho >= 0)
                                    ml_descriptor = 1./(1+exp(-cur_rho*ml_descriptor));
                                end
                                descriptor = ml_descriptor;
                                
                                
                            otherwise
                                fprintf('WRONG!!! unknown video feature type! %s\n',classification_config.feature);
                                return;
                        end
                        
                        % l2 normalize
%                         descriptor = descriptor/norm(descriptor);
                        
                        i_row = i_row +1;
                        test_data(i_row,:) = descriptor;
                        
                    end
                    
                end
                
                
                if classification_config.using_sparseLDA
                    train_data = train_data(:,sparse_weight(kernelmat_num+1,1:153));                    
                    test_data = test_data(:,sparse_weight(kernelmat_num+1,1:153));
                    
%                     train_data = train_data(:,1:153);
%                     test_data = test_data(:,1:153);
                    %sparseBase = repmat(sparse_base{kernelmat_num+1},24,1);
                    
%                     train_data = train_data(:,1:153)*sparse_base{kernelmat_num+1};
%                     test_data = test_data(:,1:153)*sparse_base{kernelmat_num+1};
                    
                end
                
%                 for i_row = 1:size(train_data,1)
%                     norm_row = norm(train_data(i_row,:));
%                     if norm_row > 1e-6
%                         train_data(i_row,:) = train_data(i_row,:)/norm_row;
%                     end
%                 end
%                 
%                 for i_row = 1:size(test_data,1)
%                     norm_row = norm(test_data(i_row,:));
%                     if norm_row > 1e-6
%                         test_data(i_row,:) = test_data(i_row,:)/norm_row;
%                     end
%                 end
                
                
                
                train_kernel_mat = train_data*train_data';
                test_kernel_mat = test_data*train_data';
                
                kernelmat_num = kernelmat_num +1;
                tot_weight = tot_weight+cur_weight;
                total_train_kernel_mat = total_train_kernel_mat + train_kernel_mat*cur_weight;
                total_test_kernel_mat = total_test_kernel_mat + test_kernel_mat*cur_weight;
                
                save_name = fullfile(classification_config.path,sprintf('kernel_mat_train_s%02d_%s_%s.mat', cur_splits,cur_descriptor,cur_feature));
                save(save_name,'train_kernel_mat');
                
                save_name = fullfile(classification_config.path,sprintf('kernel_mat_test_s%02d_%s_%s.mat', cur_splits,cur_descriptor,cur_feature));
                save(save_name,'test_kernel_mat');
                
            end % i_feature = 1:length(feature_types)
            
            
            
            
            
            
            
        end % i_desc = 1:length(descriptor_type)
        
    end % if ~classification_config.recompute_kernelmat
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % classification
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    total_train_kernel_mat = total_train_kernel_mat/tot_weight;
    total_test_kernel_mat = total_test_kernel_mat/tot_weight;
    
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
    
    cumsum_train_num_each_class = cumsum(train_num_each_class);
    cumsum_train_num_each_class = [0,cumsum_train_num_each_class];
    for i = 1:length(train_num_each_class)
        train_each_acc(i) = sum(train_pre_idx(cumsum_train_num_each_class(i)+1:cumsum_train_num_each_class(i+1)) ==...
            train_label(cumsum_train_num_each_class(i)+1:cumsum_train_num_each_class(i+1)))/train_num_each_class(i);
    end
    % train_each_acc = sum(reshape(train_pre_idx,[],n_class) == reshape(train_label,[],n_class))/train_num_per_class;
    
    [~,sortIdx] = sort(test_pre_vals,2,'descend');
    test_pre_idx = sortIdx(:,1);
    test_total_acc = sum(test_pre_idx == test_label)/length(test_label);
    
    cumsum_test_num_each_class = cumsum(test_num_each_class);
    cumsum_test_num_each_class = [0,cumsum_test_num_each_class];
    for i = 1:length(test_num_each_class)
        test_each_acc(i) = sum(test_pre_idx(cumsum_test_num_each_class(i)+1:cumsum_test_num_each_class(i+1)) ==...
            test_label(cumsum_test_num_each_class(i)+1:cumsum_test_num_each_class(i+1)))/test_num_each_class(i);
    end
    
    train_total_acc
    train_each_acc
    test_total_acc
    test_each_acc
    
    % compute confusion matrix
    confusion_matrixt = compute_confusion_matrix(test_pre_idx,test_num_each_class,class_names);
    
    save_name = fullfile(classification_config.path,sprintf('s%02d_results.mat',cur_splits) );
    save(save_name,'train_total_acc','train_each_acc','test_total_acc','test_each_acc','confusion_matrixt','class_names');
    
end % i_splits = 1:length(splits)


end


