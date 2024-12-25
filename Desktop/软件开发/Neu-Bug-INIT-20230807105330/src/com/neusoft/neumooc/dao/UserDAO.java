package com.neusoft.neumooc.dao;

import com.neusoft.core.common.exception.DAOException;
import com.neusoft.neumooc.vo.UserVO;

/**
 * @ClassName: UserDAO
 * @Description: 用户管理数据操作接口
 * @date 2016年7月13日
 */
public interface UserDAO {

	/**
	 * @Title: getUserByUserName
	 * @Description: 用户信息查询 根据用户名
	 * @param userName
	 *            用户名
	 * @return UserVO 用户信息
	 * @throws DAOException
	 */
	public UserVO getUserByUserName(String userName) throws DAOException;

}
