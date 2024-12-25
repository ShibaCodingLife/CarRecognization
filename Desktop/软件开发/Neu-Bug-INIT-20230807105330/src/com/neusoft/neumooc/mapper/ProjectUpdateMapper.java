package com.neusoft.core.common.mapper;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Update;

@Mapper
public interface ProjectUpdateMapper {

    @Update("UPDATE project " +
            "SET project_keywords = #{projectKeywords}, " +
            "    project_description = #{projectDescription}, " +
            "    project_owner = #{projectOwner} " +
            "WHERE project_name = #{projectName}")
    int updateProject(
            @Param("projectName") String projectName,
            @Param("projectKeywords") String projectKeywords,
            @Param("projectDescription") String projectDescription,
            @Param("projectOwner") String projectOwner
    );
}
