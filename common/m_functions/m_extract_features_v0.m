%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 9/6/2013
% wyw @MSRA


% call the dense trajectory feature to get video dense trajectory features,
% input: dataset_info.mat
% output: videos feature mat @global_config.extract_features.path
%               c%03d_v%03d_info.mat: [width, height, len, dt_number]
%               c%03d_v%03d_location.mat [x,y,t,....]
%               c%03d_v%03d_trajectory.mat [x_1,y_1,x_2,y_2,...,x_dt_len+1,y_dt_len+1]
%               c%03d_v%03d_features_shape.mat
%               c%03d_v%03d_features_hog.mat
%               c%03d_v%03d_features_hof.mat
%               c%03d_v%03d_features_mbhx.mat
%               c%03d_v%03d_features_mbhy.mat
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function return_state = m_extract_features_v0(global_config)
return_state =1;
% load data set info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx','train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);

% extract feature
tic;

do_class = global_config.extract_features.class_idx;
do_vids = cell(class_num,1);
switch class(do_class)
    case 'cell'
        for i = 1:length(do_class)
            do_vids{do_class{i}(1)} = [do_vids{do_class{i}(1)};do_class{i}(2)];
        end
    case 'double'
        for i = 1:length(do_class)
            do_vids{do_class(i)} = 1:vid_nums_in_class(do_class(i));
        end
        
    case 'char'
        for i = 1:class_num
            do_vids{i} = 1:vid_nums_in_class(i);
        end
        
    otherwise
        fprintf('wrong type of global_config.extract_features.class_idx!\n');
        return_state = 0;
        return;
end



for i_class = 1:class_num
    cur_class = i_class;
    
    if length(do_vids{i_class}) == 0
        continue;
    end 
    cur_class_vids = do_vids{i_class};
    if global_config.num_core > 1
        parfor i_vid = 1: length(cur_class_vids)
            do_extract_features_v0(global_config.extract_features.extract_fea_fun,fullfile(global_config.dataset_path,...
                vid_paths{cur_class}{cur_class_vids(i_vid)}),cur_class,cur_class_vids(i_vid),global_config.extract_features.path);
        end
    else
        for i_vid = 1: length(cur_class_vids)
            do_extract_features_v0(global_config.extract_features.extract_fea_fun,fullfile(global_config.dataset_path,...
                vid_paths{cur_class}{cur_class_vids(i_vid)}),cur_class,cur_class_vids(i_vid),global_config.extract_features.path);
        end
    end
    toc;
end
        
  

end

%%
% wyw 9/6/2013 @MSRA
% call extract_fea_fun to extract and save features
% graph cut all the trajectories, get the trajectory clusters
% input: trajectory and location
% output: clusters_c%03d_v%03d.mat every row is an cluster reslults, and
% the location of the center of the cluster

% v1:support bbox file
function do_extract_features_v0(extract_fea_fun,video_name,i_class,i_vid,feature_path)

if ~exist(extract_fea_fun,'file')
    fprintf('%s did not exist! \n',extract_fea_fun);
    return;
end


fprintf('class:%2d file:%3d ',i_class,i_vid);


temp_feature_name = sprintf('temp_feature_c%03d_v%03d',i_class,i_vid);
temp_vid_info_name = sprintf('temp_vid_info_c%03d_v%03d',i_class,i_vid);

% call exe to get temp file
cmd_str = [extract_fea_fun ' '  video_name ' ' temp_feature_name ' ' temp_vid_info_name];
dos(cmd_str);

% read from temp file
[dt_features,vid_info] = read_dt_data_mex(temp_feature_name,temp_vid_info_name);
% delete temp file
delete(temp_feature_name);
delete(temp_vid_info_name);


dt_features = dt_features';


if sum(sum(isnan(dt_features)))
    fprintf('wrong!!!!!!!!!!!!! %s \n',video_name);
    
    valid_idx = find(sum(isnan(dt_features),2)==0);
    
    if size(dt_features,1) - length(valid_idx) < 10
        dt_features = dt_features(valid_idx,:);
        fprintf('remove invalid dt features, continue\n');
        
    else
        return;
    end
    
    
else
    fprintf('dt_num:%d %s \n',size(dt_features,1),video_name);
end

% save features
% [width, height, len, location_dim, trajectory_dim, shape_dim, hog_dim, hof_dim, mbhx_dim, mbhy_dim]
fea_dim = vid_info(4:end);
fea_dim = cumsum(fea_dim);


% [width, height, len, dt_number]
vid_info = vid_info(1:3);
vid_info = [vid_info;size(dt_features,1)];

save_name = sprintf('%s\\c%03d_v%03d_info.mat',feature_path,i_class,i_vid);
save(save_name,'vid_info');

location = dt_features(:,1:fea_dim(1));
save_name = sprintf('%s\\c%03d_v%03d_location.mat',feature_path,i_class,i_vid);
save(save_name,'location');

trajectory = dt_features(:,fea_dim(1)+1:fea_dim(2));
save_name = sprintf('%s\\c%03d_v%03d_trajectory.mat',feature_path,i_class,i_vid);
save(save_name,'trajectory');

features = cell(length(fea_dim)-2,1);
features_shape = dt_features(:,fea_dim(2)+1:fea_dim(3));
features_hog = dt_features(:,fea_dim(3)+1:fea_dim(4));
features_hof = dt_features(:,fea_dim(4)+1:fea_dim(5));
features_mbhx = dt_features(:,fea_dim(5)+1:fea_dim(6));
features_mbhy = dt_features(:,fea_dim(6)+1:fea_dim(7));


save_name = sprintf('%s\\c%03d_v%03d_features_shape.mat',feature_path,i_class,i_vid);
save(save_name,'features_shape');

save_name = sprintf('%s\\c%03d_v%03d_features_hog.mat',feature_path,i_class,i_vid);
save(save_name,'features_hog');

save_name = sprintf('%s\\c%03d_v%03d_features_hof.mat',feature_path,i_class,i_vid);
save(save_name,'features_hof');

save_name = sprintf('%s\\c%03d_v%03d_features_mbhx.mat',feature_path,i_class,i_vid);
save(save_name,'features_mbhx');

save_name = sprintf('%s\\c%03d_v%03d_features_mbhy.mat',feature_path,i_class,i_vid);
save(save_name,'features_mbhy');

end