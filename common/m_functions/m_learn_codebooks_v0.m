% wyw 9/6/2013 @MSRA
% learn the codebook, get codebooks
% input: data_set_info.mat
%        c%03d_v%03d_features.mat
% output: codebooks_s%02d_[feature_types].mat @global_config.learn_codebooks.path
% 
function return_state = m_learn_codebooks_v0(global_config)
return_state = 1;

% load dataset info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx','train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);

learn_codebooks_config = global_config.learn_codebooks;

% get the number of dt trajecries in each videos
features_num = {};
for i_class = 1:class_num
    
    cur_class_feature_num = zeros(vid_nums_in_class(i_class),1);
    for i_vid = 1:vid_nums_in_class(i_class)
        
        load_name = fullfile(global_config.extract_features.path,sprintf('c%03d_v%03d_info.mat',i_class,i_vid));
        load(load_name,'vid_info');
        cur_class_feature_num(i_vid) = vid_info(4);
        
    end
    
    features_num{i_class} = cur_class_feature_num;
end

switch learn_codebooks_config.params.method
    case 'vl_kmeans' % specify the method
        
        % generate the splits and feature types according to global_config
        
        split_feature_type_params = cell(length(learn_codebooks_config.feature_types)*length(learn_codebooks_config.splits),2);
        
        splits = learn_codebooks_config.splits;
        feature_types = learn_codebooks_config.feature_types;
        
        i_param = 0;
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
                        
            for i_feature = 1:length(feature_types)
                cur_feature = feature_types{i_feature};
                
                i_param = i_param+1;
                split_feature_type_params{i_param,1} = cur_splits;
                split_feature_type_params{i_param,2} = cur_feature;
                               
            end
            
        end % iSplit
        
        
        vl_setup;
        
      
        
        if global_config.num_core > 1
            parfor i_param = 1:size(split_feature_type_params,1)
                do_learn_codebook_vl_kmeans(split_feature_type_params{i_param,1},split_feature_type_params{i_param,2},...
                    features_num,global_config);
            end
            
        else
            for i_param = 1:size(split_feature_type_params,1)
                if ~do_learn_codebook_vl_kmeans(split_feature_type_params{i_param,1},split_feature_type_params{i_param,2},...
                        features_num,global_config)
                    return_state = 0;
                    return;
                end
            end
        end
            
          
        
        
    otherwise
        fprintf('wrong global_config.learn_codebooks.params.method %s\n',global_config.learn_codebooks.params.method);
        return_state = 0;
        return;
        
        
end



end

%%
function return_state = do_learn_codebook_vl_kmeans(cur_splits,cur_feature,features_num,global_config)
return_state = 1;
fprintf('learn codebook of split:%d feature:%s\n',cur_splits,cur_feature);
load(global_config.read_dataset_info.file_name);
learn_codebooks_config = global_config.learn_codebooks;

if (learn_codebooks_config.params.re_select_data) ||...
        ~ exist(fullfile(global_config.learn_codebooks.path,sprintf('select_data_%02d_%s.mat',cur_splits,cur_feature)),'file')
    
    % seletct data
    select_data = [];
    
    for i_class = 1:class_num
        
        dt_features_num = features_num{i_class};
        
        
        train_set_dt_feature_num = dt_features_num(train_set_idx{cur_splits}{i_class});
        
        endIdx_each_vid = cumsum(train_set_dt_feature_num);
        
        selectIdx_each_vid = cell(length(train_set_dt_feature_num),1);
        total_dt_feature_num_curClass = sum(train_set_dt_feature_num);
        
        
        if total_dt_feature_num_curClass > learn_codebooks_config.params.select_data_num_each_class
            select_idx = randperm(total_dt_feature_num_curClass,learn_codebooks_config.params.select_data_num_each_class);
            
            selectIdx_each_vid{1} = select_idx(find(select_idx<=endIdx_each_vid(1)));
            
            for i_vid = 2:length(train_set_dt_feature_num)
                selectIdx_each_vid{i_vid} = select_idx(find(select_idx>endIdx_each_vid(i_vid-1) & select_idx<=endIdx_each_vid(i_vid)))...
                    -endIdx_each_vid(i_vid-1);
                
            end
        else
            for i_vid = 1:length(train_set_dt_feature_num)
                selectIdx_each_vid{i_vid} = (1:train_set_dt_feature_num(i_vid))';
            end
        end
        
        
        for i_vid = 1:length(train_set_dt_feature_num)
            
%             fprintf('loading %03d,%03d  ',i_class,train_set_idx{cur_splits}{i_class}(i_vid));
            load_name = fullfile(global_config.extract_features.path,sprintf('c%03d_v%03d_features_%s.mat',i_class,train_set_idx{cur_splits}{i_class}(i_vid),cur_feature));
            load(load_name); %load variable dt_features
            
            switch cur_feature
                case 'shape'
                    features = features_shape;
                case 'hog'
                    features = features_hog;
                case 'hof'
                    features = features_hof;
                case 'mbhx'
                    features = features_mbhx;
                case 'mbhy'
                    features = features_mbhy;
                otherwise
                    fprintf('wrong feature type %s\n',cur_feature);
                    return_state = 0;
                    return;
            end
            
            
            if length(selectIdx_each_vid{i_vid}) > 0
                select_data = [select_data;features(selectIdx_each_vid{i_vid},:)];
            end
            
            clear features;
            
           
            
        end % ivid ends
         fprintf('splits:%02d feature %s select num:%d \n',cur_splits,cur_feature,size(select_data,1));
    end % iclass ends
    
    
    if size(select_data,1) > learn_codebooks_config.params.select_data_num_total
        select_data = select_data(randperm(size(select_data,1),learn_codebooks_config.params.select_data_num_total),:);
    end
    
    
    save_name = fullfile(global_config.learn_codebooks.path,sprintf('select_data_s%02d_%s.mat',cur_splits,cur_feature));
    save(save_name,'select_data');
else
    loadName = fullfile(global_config.learn_codebooks.path,sprintf('select_data_s%02d_%s.mat',cur_splits,cur_feature));
    load(loadName);
end

% vl_kmeans
fprintf('kmeans of split:%d feature:%s\n',cur_splits,cur_feature);
codebooks = vl_kmeans(select_data',learn_codebooks_config.params.codebook_size,'numRepetitions',8,'algorithm', 'elkan')';

save_name = fullfile(learn_codebooks_config.path,sprintf('codebooks_s%02d_%s.mat',cur_splits,cur_feature));
save(save_name,'codebooks');
clear codebooks;
end