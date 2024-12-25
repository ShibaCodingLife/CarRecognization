package com.neusoft.core.common.service.impl;

import com.neusoft.core.common.ProjectUpdateRequest;
import com.neusoft.core.common.service.ProjectUpdateService;
import com.neusoft.core.common.mapper.ProjectUpdateMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class ProjectUpdateServiceImpl implements ProjectUpdateService {

    @Autowired
    private ProjectUpdateMapper projectUpdateMapper;

    @Override
    public boolean updateProject(Project request) {
        int rowsAffected = projectUpdateMapper.updateProject(
                request.getProjectName(),
                request.getProjectKeywords(),
                request.getProjectDescription(),
                request.getProjectOwner()
        );
        return rowsAffected > 0; // 返回是否更新成功
    }
}
