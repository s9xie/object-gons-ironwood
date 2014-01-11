% wyw 9/6/2013 @MSRA
% inputs: features and codebooks
% outputs: codes_s%02d_c%03d_v%03d_[feature_type].mat

function return_state = m_coding_v0(global_config)
return_state = 1;

% load dataset info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx','train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);

coding_config = global_config.coding;

% generate the splits and feature types according to global_config

splits = coding_config.splits;
feature_types = coding_config.feature_types;


do_sets = global_config.coding.class_idx;

do_vids = cell(length(splits),class_num);
switch class(do_sets)
    case 'cell'
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:length(do_sets)
                do_vids{cur_splits,do_sets{i}(1)} = [do_vids{cur_splits,do_sets{i}(1)};do_sets{i}(2)];
            end
            
        end
    case 'double'
        
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);          
            for i = 1:length(do_sets)
                do_vids{cur_splits,do_sets(i)} = [train_set_idx{cur_splits}{do_sets(i)};test_set_idx{cur_splits}{do_sets(i)}];
            end
        end
        
    case 'char'
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:class_num
                do_vids{cur_splits,i} = [train_set_idx{cur_splits}{i};test_set_idx{cur_splits}{i}];
            end
        end
        
    otherwise
        fprintf('wrong type of global_config.extract_features.class_idx!\n');
        return_state = 0;
        return;
end

for i_splits = 1:length(splits)
    cur_splits = splits(i_splits);
    
    for i_class = 1:class_num
       
        cur_class = i_class;
        if length(do_vids{cur_splits,i_class}) == 0
            continue;
        end
        
        for i_feature = 1:length(feature_types)
            cur_feature = feature_types{i_feature};
            
            % load the related codebook
            load_name = fullfile(global_config.learn_codebooks.path,sprintf('codebooks_s%02d_%s.mat',cur_splits,cur_feature) );
            load(load_name,'codebooks');
            
            cur_codebook = codebooks;
            % all data
            fprintf('coding splits:%02d  c:%03d feature:%s\n',cur_splits,cur_class,cur_feature);
            tic;
            cur_vid_idx = do_vids{cur_splits,i_class};
            
            switch coding_config.params.method
                case 'LSAQ' % specify the method
                    
                    if global_config.num_core > 1
                        parfor i_vid = 1:length(cur_vid_idx)
                            do_coding_LSAQ(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),cur_codebook,coding_config,global_config.extract_features.path);
                        end
                    else
                        for i_vid = 1:length(cur_vid_idx)
                            do_coding_LSAQ(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),cur_codebook,coding_config,global_config.extract_features.path);
                        end
                    end
                    
                otherwise % other method
                    
                    fprintf('wrong global_config.learn_codebooks.params.method %s\n',global_config.learn_codebooks.params.method);
                    return_state = 0;
                    return;
                    
                    
            end % switch end
            toc;
        end  % ifeature_type
    end % iclass ends
    
end % i_split








end % function end
%%
function do_coding_LSAQ(cur_splits,cur_feature,i_class,i_vid,codebook,coding_config,feature_path)


KNN = coding_config.params.KNN;
beta = coding_config.params.beta;

% load the feature
load_name = fullfile(feature_path,sprintf('c%03d_v%03d_features_%s.mat',i_class,i_vid,cur_feature) );
load(load_name);

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
        return;
end

codes = do_coding_LSAQ_mex(features',codebook',KNN,beta)';

if sum(sum(isnan(codes)))
   fprintf('NaN data occur!\n');
   return;
end

codes = sparse(codes);

save_name = fullfile(coding_config.path,sprintf('codes_s%02d_c%03d_v%03d_%s.mat',cur_splits,i_class,i_vid,cur_feature));
save(save_name,'codes');

end
