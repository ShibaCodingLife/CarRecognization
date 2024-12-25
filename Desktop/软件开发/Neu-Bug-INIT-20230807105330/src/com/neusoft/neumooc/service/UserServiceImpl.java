package com.neusoft.neumooc.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.neusoft.core.common.BaseService;
import com.neusoft.core.common.exception.DAOException;
import com.neusoft.core.common.exception.ServiceException;
import com.neusoft.neumooc.dao.UserDAO;
import com.neusoft.neumooc.vo.UserVO;

/**
 * @ClassName: UserServiceImpl
 * @Description: 用户管理业务逻辑处理类
 * @date 2016年7月13日
 */
@Service
@Transactional
public class UserServiceImpl extends BaseService implements UserService {

	// 用户管理数据操作类
	@Autowired
	private UserDAO userDAO;

	/**
	 * @Title: getUserByUserName
	 * @Description: 用户信息查询 根据用户名
	 * @param userName
	 *            用户名
	 * @return 用户信息
	 */
	@Override
	public UserVO getUserByUserName(String userName) {
		UserVO vo = null;
		try {
			vo = userDAO.getUserByUserName(userName);
		} catch (DAOException e) {
			throw new ServiceException(
					"class UserDAOImpl-->method getUserByUserName error", e);
		} catch (Exception e) {
			throw new ServiceException(
					"class UserServiceImpl-->method getUserByUserName error", e);
		}
		return vo;
	}
}
