package com.neusoft.neumooc.service;

import com.neusoft.neumooc.vo.UserVO;

/**
 * @ClassName: UserService
 * @Description: 用户管理业务逻辑处理接口
 * @date 2016年7月13日
 */
public interface UserService {

	/**
	 * @Title: getUserByUserName
	 * @Description: 用户信息查询 根据用户名
	 * @param userName
	 *            用户名
	 * @return 用户信息
	 */
	public UserVO getUserByUserName(String userName);

}
