package com.neusoft.core.common;

import com.neusoft.core.common.exception.DAOException;
import com.neusoft.neumooc.vo.UserVO;

public class Project {

    private String projectName;
    private String projectKeywords;
    private String projectDescription;
    private String projectOwner;

    // Getters and Setters
    public String getProjectName() {
        return projectName;
    }

    public void setProjectName(String projectName) {
        this.projectName = projectName;
    }

    public String getProjectKeywords() {
        return projectKeywords;
    }

    public void setProjectKeywords(String projectKeywords) {
        this.projectKeywords = projectKeywords;
    }

    public String getProjectDescription() {
        return projectDescription;
    }

    public void setProjectDescription(String projectDescription) {
        this.projectDescription = projectDescription;
    }

    public String getProjectOwner() {
        return projectOwner;
    }

    public void setProjectOwner(String projectOwner) {
        this.projectOwner = projectOwner;
    }
}
