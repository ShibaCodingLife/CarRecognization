package com.neusoft.core.common;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import com.neusoft.core.common.service.ProjectUpdateService;

/**
 * @ClassName: ProjectUpdateController(ZHJ工作)
 * @Description: 基础Controller类，用于处理项目修改请求
 */
@RestController
@RequestMapping("/api/project")
public class ProjectUpdateController {

    @Autowired
    private ProjectUpdateService projectUpdateService;

    /**
     * 更新项目数据
     *
     * @param request 项目信息
     * @return 成功或失败消息
     */
    @PostMapping("/update")
    public ResponseEntity<String> updateProject(@RequestBody Project request) {
        boolean isUpdated = projectUpdateService.updateProject(request);
        if (isUpdated) {
            return ResponseEntity.ok("项目修改成功！");
        } else {
            return ResponseEntity.status(500).body("项目修改失败！");
        }
    }
}
